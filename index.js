import express from "express";
import bodyParser from "body-parser";
import dotenv from "dotenv";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

const app = express();
app.use(bodyParser.json());

// Init OpenAI + Supabase
const llm = new ChatOpenAI({
  temperature: 0,
  modelName: "gpt-4o-mini",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

// --- Vector Search ---
async function searchDocuments(queryEmbedding, matchCount = 5, filter = {}) {
  const { data, error } = await supabase.rpc("match_documents", {
    query_embedding: queryEmbedding,
    match_count: matchCount,
    filter: filter,
  });

  if (error) {
    console.error("Error searching documents:", error);
    return [];
  }

  return data || [];
}

// --- Helper: Build context from docs ---
function buildContext(docs) {
  if (!docs || docs.length === 0) {
    return "No relevant context found.";
  }

  return docs
    .map(
      (d, i) =>
        `Document ${i + 1}:\n${d.content || ""}\n(Source: ${
          d.metadata?.source || "unknown"
        })`
    )
    .join("\n\n");
}

// --- Chat Endpoint ---
app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    // Step 1: Embed query
    const embedding = await embeddings.embedQuery(message);

    // Step 2: Search Supabase
    const results = await searchDocuments(embedding, 5, {});

    // Step 3: Build context
    const context = buildContext(results);

    // Step 4: Send to LLM
    const response = await llm.call([
      {
        role: "system",
        content: `You are an assistant with access to a knowledge base. Use the provided context to answer queries.
If context is missing, say you don't know instead of making up answers.`,
      },
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${message}`,
      },
    ]);

    // Step 5: Return
    res.json({
      answer: response?.text || "No answer",
      sources: results.map((r) => r.metadata?.source || "unknown"),
    });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// --- Healthcheck ---
app.get("/", (req, res) => {
  res.send("RAG chatbot server running ✅");
});

// --- Start Server ---
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
