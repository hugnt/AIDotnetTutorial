using Microsoft.AspNetCore.Mvc;
using Microsoft.SemanticKernel;

var builder = WebApplication.CreateBuilder(args);

// Add services and swagger
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Ensure configuration keys exist (optional: validate here)
var deploymentId = builder.Configuration["OpenAI:DeploymentId"];
var endpoint = builder.Configuration["OpenAI:Endpoint"];
var apiKey = builder.Configuration["OpenAI:ApiKey"];
if (string.IsNullOrWhiteSpace(deploymentId) ||
    string.IsNullOrWhiteSpace(endpoint) ||
    string.IsNullOrWhiteSpace(apiKey))
{
    throw new InvalidOperationException("OpenAI configuration missing. Check OpenAI:DeploymentId, OpenAI:Endpoint, OpenAI:ApiKey.");
}

// Register Azure OpenAI chat completion (this registers the ChatCompletion service)
builder.Services.AddAzureOpenAIChatCompletion(
    deploymentName: deploymentId,
    endpoint: endpoint,
    apiKey: apiKey
);

// Register Kernel as singleton using the IServiceProvider-based constructor
builder.Services.AddSingleton(sp => new Kernel(sp));

// Register your chat service
builder.Services.AddTransient<IChatService, ChatService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.MapGet("/training-sentiment-classification-model", (IChatService chatService) =>
{
    chatService.TrainAndSaveModel();
    return Results.Ok("Model trained and saved successfully.");
})
.WithName("TrainingSentimentClassificationModel");

app.MapPost("/chat-sentiment-analysis", async (IChatService chatService, [FromBody] ChatRequest chatRequest) =>
{
    var response = await chatService.GetChatSentimentAnalysis(chatRequest);
    return Results.Ok(response);
}).WithName("ChatSentimentAnalysis");

app.Run();
