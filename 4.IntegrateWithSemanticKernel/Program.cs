using Microsoft.AspNetCore.Mvc;
using Microsoft.SemanticKernel;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddKernel();
builder.Services.AddAzureOpenAIChatCompletion(
    deploymentName: builder.Configuration["OpenAI:DeploymentId"],
    endpoint: builder.Configuration["OpenAI:Endpoint"],
    apiKey: builder.Configuration["OpenAI:ApiKey"]
);
builder.Services.AddTransient((serviceProvider) =>
{
    return new Kernel(serviceProvider);
});

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

app.MapPost("/summary", async (IChatService chatService, [FromBody] ChatRequest chatRequest) =>
{
    var response = await chatService.GetChatSummaryResponseAsync(chatRequest);
    return Results.Ok(response);
})
.WithName("GetChatSummary");

app.Run();