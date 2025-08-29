import express from 'express';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { BufferMemory } from "langchain/memory";
import cors from 'cors';
import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// ✅ Allowed origins (your frontend + local dev)
const allowedOrigins = [
  "https://orange-dotterel-510746.hostingersite.com",
  "http://localhost:3000"
];

// ✅ Setup CORS properly
app.use(
  cors({
    origin: function (origin, callback) {
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error("Not allowed by CORS"));
      }
    },
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  })
);

// ✅ Parse JSON
app.use(express.json());

// ✅ Handle preflight
app.options("*", cors());

// ---- Global constants ----
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const GOOGLE_SEARCH_API_KEY = process.env.GOOGLE_SEARCH_API_KEY;
const GOOGLE_SEARCH_ENGINE_ID = process.env.GOOGLE_SEARCH_ENGINE_ID;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;

// ---- In-memory chat history ----
const memory = new BufferMemory({
  memoryKey: "chat_history",
  inputKey: "question",
  outputKey: "answer",
  returnMessages: true
});

// ---- Routes ----
app.get('/', (req, res) => {
  res.send('✅ Deepwoods AI Assistant is running!');
});

app.post('/api/chat', async (req, res) => {
  try {
    const { question, pdfUrl } = req.body;
    if (!question) {
      return res.status(400).json({ error: "Missing question" });
    }

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash-latest",
      apiKey: GOOGLE_API_KEY,
    });

    if (pdfUrl && pdfUrl !== 'general_discussion') {
      // ---- Supabase setup ----
      const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
      const embeddings = new GoogleGenerativeAIEmbeddings({
        modelName: "embedding-001",
        apiKey: GOOGLE_API_KEY,
      });

      const vectorStore = new SupabaseVectorStore(embeddings, {
        client: supabase,
        tableName: 'documents',
        queryName: 'match_documents',
      });

      const retriever = vectorStore.asRetriever();
      const retrievedDocs = await retriever.getRelevantDocuments(question);

      if (retrievedDocs && retrievedDocs.length > 0) {
        // ✅ Prompt with history + context
        const prompt = ChatPromptTemplate.fromMessages([
          ["system", "Answer the user's question using the context and history."],
          ["system", "<chat_history>{chat_history}</chat_history>"],
          ["system", "<context>{context}</context>"],
          ["human", "{input}"]
        ]);

        const documentChain = await createStuffDocumentsChain({
          llm: model,
          prompt: prompt
        });

        const retrievalChain = await createRetrievalChain({
          retriever,
          combineDocsChain: documentChain,
        });

        const result = await retrievalChain.invoke({
          input: question,
          chat_history: await memory.loadMemoryVariables({})
        });

        await memory.saveContext({ question }, { answer: result.answer });
        return res.status(200).json({ answer: result.answer });
      } else {
        // ---- Google search fallback ----
        const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_SEARCH_API_KEY}&cx=${GOOGLE_SEARCH_ENGINE_ID}&q=${encodeURIComponent(question)}`;
        const searchResponse = await axios.get(searchUrl);
        const searchResults = searchResponse.data.items || [];

        let searchContext = '';
        searchResults.forEach(item => {
          searchContext += `Title: ${item.title}\nLink: ${item.link}\nSnippet: ${item.snippet}\n\n`;
        });

        if (searchContext) {
          const prompt = ChatPromptTemplate.fromMessages([
            ["system", "You are an AI assistant that answers using search results."],
            ["system", "{context}"],
            ["human", "{question}"]
          ]);

          const chain = prompt.pipe(model);
          const result = await chain.invoke({
            context: searchContext,
            question
          });

          const warning = "⚠️ Note: No relevant docs found, so I searched the web.";
          return res.status(200).json({ answer: `${warning}\n\n${result.content}` });
        } else {
          return res.status(200).json({ answer: "I could not find a relevant answer in your documents or on the internet." });
        }
      }
    } else {
      // ---- General discussion mode ----
      const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_SEARCH_API_KEY}&cx=${GOOGLE_SEARCH_ENGINE_ID}&q=${encodeURIComponent(question)}`;
      const searchResponse = await axios.get(searchUrl);
      const searchResults = searchResponse.data.items || [];

      let searchContext = '';
      searchResults.forEach(item => {
        searchContext += `Title: ${item.title}\nLink: ${item.link}\nSnippet: ${item.snippet}\n\n`;
      });

      if (searchContext) {
        const prompt = ChatPromptTemplate.fromMessages([
          ["system", "You are an AI assistant that answers using search results."],
          ["system", "{context}"],
          ["human", "{question}"]
        ]);

        const chain = prompt.pipe(model);
        const result = await chain.invoke({
          context: searchContext,
          question
        });

        return res.status(200).json({ answer: result.content });
      } else {
        return res.status(200).json({ answer: "I could not find a relevant answer on the internet." });
      }
    }
  } catch (error) {
    console.error("❌ Error in handler:", error);
    res.status(500).json({ error: "An error occurred." });
  }
});

// ---- Start server ----
app.listen(port, () => {
  console.log(`🚀 Server listening on port ${port}`);
});
