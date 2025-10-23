using Microsoft.SemanticKernel;

public interface IChatService
{
    Task<ChatResponse> GetChatResponseAsync(ChatRequest chatRequest);
}

public class ChatService : IChatService
{
    private readonly Kernel _kernel;
    public ChatService(Kernel kernel)
    {
        _kernel = kernel;

    }

    public async Task<ChatResponse> GetChatResponseAsync(ChatRequest chatRequest)
    {
        var summarizeKernelFunction = _kernel.CreateFunctionFromPrompt(
                promptTemplate: File.ReadAllText("./Data/summarize.skprompt.txt"),
                functionName: "SummarizeText");
        return await summarizeKernelFunction.InvokeAsync(chatRequest.UserPrompt);
    }
}
