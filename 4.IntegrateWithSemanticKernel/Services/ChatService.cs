using Microsoft.SemanticKernel;

public interface IChatService
{
    Task<ChatResponse> GetChatSummaryResponseAsync(ChatRequest chatRequest);
}

public class ChatService : IChatService
{
    private readonly Kernel _kernel;
    public ChatService(Kernel kernel)
    {
        _kernel = kernel;

    }

    public async Task<ChatResponse> GetChatSummaryResponseAsync(ChatRequest chatRequest)
    {
        var summarizeKernelFunction = _kernel.CreateFunctionFromPrompt(
                promptTemplate: File.ReadAllText("./Data/summarize.skprompt.txt"),
                functionName: "SummarizeText");
        var result = await _kernel.InvokeAsync(summarizeKernelFunction, new() { ["input"] = chatRequest.UserPrompt });
        return new ChatResponse { Role = "Assistant", Content = result.GetValue<string>() };
    }
}