public class ChatRequest
{
    public string UserPrompt { get; set; }
}


public class ChatResponse
{
    public string Role { get; set; }
    public string Content { get; set; }
}