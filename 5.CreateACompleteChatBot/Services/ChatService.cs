using Microsoft.ML;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

public interface IChatService
{
    void TrainAndSaveModel();
    Task<ChatResponse> GetChatSentimentAnalysis(ChatRequest chatRequest);
}

public class ChatService : IChatService
{
    private readonly Kernel _kernel;
    private readonly IChatCompletionService _chatCompletionService;
    private readonly ChatHistory _chatHistory;
    private const string systemPrompt = """
        You are an empathetic assistant that reads input in the exact format:
        Sentiment: 'Positive' | 'Negative'. Content: <user message>

        RULES:
        - Trust the provided Sentiment value ("Positive" or "Negative") as the primary context, and use the Content only to add nuance (cause, intensity, or details).
        - Return EXACTLY ONE short, natural-sounding paragraph in English (1–3 sentences). That paragraph must:
        1) Briefly acknowledge or analyze the user's emotional state; and
        2) Offer one practical, compassionate piece of advice tailored to that emotion.
        - Do NOT output the labels "Positive"/"Negative" or any JSON, bullet list, or extra metadata. Do not include any internal reasoning or chain-of-thought.
        - Tone: human, warm, concise, non-judgmental. Avoid emojis and technical phrasing. Avoid generic platitudes—make the advice actionable.
        - If Sentiment is missing or malformed, infer the emotion from Content and still return one short paragraph.
        - If Content is empty, return one short sentence asking for clarification (e.g., "Could you tell me more about how you're feeling?").
        """;

    public ChatService(Kernel kernel)
    {
        _kernel = kernel;
        _chatCompletionService = _kernel.GetRequiredService<IChatCompletionService>();

        _chatHistory = new ChatHistory();

    }

    public void TrainAndSaveModel()
    {
        var mlContext = new MLContext();
        var data = mlContext.Data.LoadFromTextFile<SentimentData>("./Data/data.tsv", hasHeader: true);
        var pipeline = mlContext.Transforms.Text
            .FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
        var model = pipeline.Fit(data);
        mlContext.Model.Save(model, data.Schema, "./ExportedModels/SentimentModel.zip");
    }

    public async Task<ChatResponse> GetChatSentimentAnalysis(ChatRequest chatRequest)
    {
        //Classify the sentiment of the user prompt
        var mlContext = new MLContext();
        ITransformer predictionPipeline = mlContext.Model.Load("./ExportedModels/SentimentModel.zip", out var modelInputSchema);
        var predEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(predictionPipeline);
        var sentimentData = new SentimentData { Text = chatRequest.UserPrompt };
        var prediction = predEngine.Predict(sentimentData);
        var sentiment = prediction.Prediction ? "Positive" : "Negative";

        //Analyze the sentiment using Semantic Kernel and return the response
        var userMessage = $"Sentiment: '{sentiment}'. Content: {chatRequest.UserPrompt}";
        _chatHistory.AddUserMessage(userMessage);
        var execSettings = new PromptExecutionSettings()
        {

        };
        // Get the response from the AI
        var result = await _chatCompletionService.GetChatMessageContentAsync(
            _chatHistory,
            executionSettings: execSettings,
            kernel: _kernel);
        return new ChatResponse
        {
            Role = "system",
            Content = result.Content ?? string.Empty
        };
    }
}