using System.Text;
using Microsoft.Extensions.Options;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<OpenAIOptions>(builder.Configuration.GetSection("OpenAI"));
builder.Services.Configure<QdrantOptions>(builder.Configuration.GetSection("Qdrant"));

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Qdrant client
builder.Services.AddSingleton(sp =>
{
    var q = sp.GetRequiredService<IOptions<QdrantOptions>>().Value;
    return new QdrantClient(new Uri(q.Url));
});

// OpenAI client (optional)
builder.Services.AddSingleton(sp =>
{
    var o = sp.GetRequiredService<IOptions<OpenAIOptions>>().Value;
    return string.IsNullOrWhiteSpace(o.ApiKey) ? null : new OpenAIClient(o.ApiKey);
});

var app = builder.Build();
app.UseSwagger();
app.UseSwaggerUI();

app.MapPost("/index", async (IOptions<QdrantOptions> qOpt, QdrantClient qdrant) =>
{
    var q = qOpt.Value;
    await EnsureCollectionAsync(qdrant, q.Collection, q.VectorSize);

    var projectRoot = Directory.GetParent(app.Environment.ContentRootPath)!.FullName;
    var docsDir = Path.Combine(projectRoot, "docs");
    Console.WriteLine("Error:" + docsDir);


    if (!Directory.Exists(docsDir))
        return Results.BadRequest(new { ok = false, error = "docs folder not found" });

    var files = Directory.GetFiles(docsDir)
        .Where(f => f.EndsWith(".md", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".txt", StringComparison.OrdinalIgnoreCase))
        .ToArray();

    // NOTE: embeddings required for real vector search. If no OpenAI key, we still store payload but vector will be zeros.
    // Better: add key later and re-index.
    var openAi = app.Services.GetService<OpenAIClient>();
    var embedModel = app.Services.GetRequiredService<IOptions<OpenAIOptions>>().Value.EmbeddingModel;

    ulong id = (ulong)DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

    foreach (var file in files)
    {
        var content = await File.ReadAllTextAsync(file);
        var chunks = ChunkText(content, maxChars: 900);

        for (int i = 0; i < chunks.Count; i++)
        {
            var chunk = chunks[i];

            float[] vector;
            if (openAi is null)
            {
                vector = new float[q.VectorSize]; // placeholder if no key
            }
            else
            {
                vector = await EmbedAsync(openAi, embedModel, chunk);
            }

            var payload = new Dictionary<string, Value>
            {
                ["file"] = Path.GetFileName(file),
                ["chunkIndex"] = i,
                ["text"] = chunk
            };

            await qdrant.UpsertAsync(
                collectionName: q.Collection,
                points: new[]
                {
                    new PointStruct
                    {
                        Id = new PointId { Num = id++ },
                        Vectors = vector,
                        Payload = { payload }
                    }
                }
            );
        }
    }

    return Results.Ok(new { ok = true, indexedFiles = files.Length, note = openAi is null ? "Indexed with zero-vectors (no OpenAI key). Re-index later for real search." : null });
});

app.MapPost("/ask", async (AskRequest req,
    IOptions<QdrantOptions> qOpt,
    IOptions<OpenAIOptions> oOpt,
    QdrantClient qdrant,
    OpenAIClient? openAi) =>
{
    if (string.IsNullOrWhiteSpace(req.Question))
        return Results.BadRequest(new { ok = false, error = "question is required" });

    var q = qOpt.Value;
    var o = oOpt.Value;

    await EnsureCollectionAsync(qdrant, q.Collection, q.VectorSize);

    // If no embeddings available, we fallback: return most recent chunks (not semantic)
    List<ScoredPoint> hits;
    if (openAi is null)
    {
        // crude fallback: just scroll some points
        // Fallback: no embeddings → scroll
        var scroll = await qdrant.ScrollAsync(
            collectionName: q.Collection,
            limit: 5
        );

        hits = scroll.Result?
            .Select(rp => ToScoredPoint(rp))
            .ToList()
            ?? new List<ScoredPoint>();
    }
    else
    {
        var qVec = await EmbedAsync(openAi, o.EmbeddingModel, req.Question);

        var search = await qdrant.SearchAsync(
            collectionName: q.Collection,
            vector: qVec,
            limit: 5
        );

        hits = search.ToList();
    }

    var contexts = BuildContexts(hits);

    // If no LLM key -> simple response
    if (openAi is null)
    {
        return Results.Ok(new
        {
            ok = true,
            answer = "No LLM key configured. Here are the most relevant policy snippets I have (or recent snippets).",
            sources = hits.Select(ToSource).ToList(),
            snippets = contexts
        });
    }

    var system = """
You are an internal company policy assistant.
Rules:
- Answer ONLY using the provided sources.
- If not found in sources, say: "I couldn't find this in the available policies."
- Keep it short and clear.
- At the end, list sources as: (file, chunk).
""";

    var chat = openAi.GetChatClient(o.ChatModel);
    var messages = new List<ChatMessage>
    {
        new SystemChatMessage(system),
        new UserChatMessage($"Question: {req.Question}\n\nSOURCES:\n{contexts}")
    };

    var resp = await chat.CompleteChatAsync(messages, new ChatCompletionOptions { Temperature = 0.2f });
    var answer = resp.Value.Content[0].Text;

    return Results.Ok(new
    {
        ok = true,
        answer,
        sources = hits.Select(ToSource).ToList()
    });
});

app.Run();

static async Task EnsureCollectionAsync(QdrantClient qdrant, string collection, int vectorSize)
{
    var exists = await qdrant.CollectionExistsAsync(collection);
    if (!exists)
    {
        await qdrant.CreateCollectionAsync(collection, new VectorParams
        {
            Size = (uint)vectorSize,
            Distance = Distance.Cosine
        });
    }
}

static List<string> ChunkText(string text, int maxChars)
{
    var parts = text
        .Split(new[] { "\r\n\r\n", "\n\n" }, StringSplitOptions.RemoveEmptyEntries)
        .Select(s => s.Trim())
        .Where(s => s.Length > 0)
        .ToList();

    var chunks = new List<string>();
    var sb = new StringBuilder();

    foreach (var p in parts)
    {
        if (sb.Length + p.Length + 2 > maxChars)
        {
            if (sb.Length > 0) chunks.Add(sb.ToString());
            sb.Clear();
        }
        if (sb.Length > 0) sb.Append("\n");
        sb.Append(p);
    }

    if (sb.Length > 0) chunks.Add(sb.ToString());
    return chunks;
}

static string BuildContexts(List<ScoredPoint> hits)
{
    var sb = new StringBuilder();
    for (int i = 0; i < hits.Count; i++)
    {
        var p = hits[i].Payload;
        var file = p.TryGetValue("file", out var fv) ? fv.StringValue : "unknown";
        var chunkIndex = p.TryGetValue("chunkIndex", out var cv) ? cv.IntegerValue.ToString() : "?";
        var text = p.TryGetValue("text", out var tv) ? tv.StringValue : "";

        sb.AppendLine($"Source {i + 1} | file: {file} | chunk: {chunkIndex}");
        sb.AppendLine(text);
        sb.AppendLine("\n---\n");
    }
    return sb.ToString();
}

static object ToSource(ScoredPoint sp)
{
    var p = sp.Payload;
    return new
    {
        file = p.TryGetValue("file", out var fv) ? fv.StringValue : "unknown",
        chunkIndex = p.TryGetValue("chunkIndex", out var cv) ? cv.IntegerValue : 0,
        score = sp.Score
    };
}

static async Task<float[]> EmbedAsync(OpenAIClient openAi, string model, string input)
{
    var embeddings = openAi.GetEmbeddingClient(model);
    var res = await embeddings.GenerateEmbeddingAsync(input);
    // OpenAI .NET returns ReadOnlyMemory<float> typically:
    return res.Value.ToFloats().ToArray();
}
static ScoredPoint ToScoredPoint(RetrievedPoint rp, float score = 1.0f)
{
    return new ScoredPoint
    {
        Id = rp.Id,
        Payload = { rp.Payload },
        Vectors = rp.Vectors,
        Score = score
    };
}

public record AskRequest(string Question);

public class OpenAIOptions
{
    public string ApiKey { get; set; } = "";
    public string ChatModel { get; set; } = "gpt-4o-mini";
    public string EmbeddingModel { get; set; } = "text-embedding-3-small";
}

public class QdrantOptions
{
    public string Url { get; set; } = "http://localhost:6333";
    public string Collection { get; set; } = "company_policies";
    public int VectorSize { get; set; } = 1536;
}

