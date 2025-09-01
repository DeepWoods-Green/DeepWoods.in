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
app.use(express.json());
app.use(cors());

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const GOOGLE_SEARCH_API_KEY = process.env.GOOGLE_SEARCH_API_KEY;
const GOOGLE_SEARCH_ENGINE_ID = process.env.GOOGLE_SEARCH_ENGINE_ID;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;

const memory = new BufferMemory({
  memoryKey: "chat_history",
  inputKey: "question",
  outputKey: "answer",
  returnMessages: true
});

app.get('/', (req, res) => {
  res.send('Welcome to the Deepwoods AI Assistant!');
});

app.post('/api/chat', async (req, res) => {
  console.log("\n--- New Request ---");
  try {
    const { question } = req.body;
    console.log(`[DEBUG] Input: question="${question}"`);

    if (!question) {
      console.log("[ERROR] Missing question in request body.");
      return res.status(400).json({ error: "Missing question" });
    }

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash-latest",
      apiKey: GOOGLE_API_KEY,
    });

    // SEARCH ACROSS ALL DOCUMENTS
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "embedding-001",
      apiKey: GOOGLE_API_KEY,
    });
    const vectorStore = new SupabaseVectorStore(embeddings, {
      client: supabase,
      tableName: 'documents',
      queryName: 'match_documents'
    });
    const retriever = vectorStore.asRetriever();
    const retrievedDocs = await retriever.getRelevantDocuments(question);

    if (retrievedDocs && retrievedDocs.length > 0) {
      console.log(`[DEBUG] Found ${retrievedDocs.length} relevant documents in vector store.`);
      const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Answer the user's question using the context and history.\n<chat_history>{chat_history}</chat_history>\n<context>{context}</context>"],
        ["human", "{input}"]
      ]);

      const documentChain = await createStuffDocumentsChain({ llm: model, prompt });
      const retrievalChain = await createRetrievalChain({ retriever, combineDocsChain: documentChain });
      const chatHistory = await memory.loadMemoryVariables({});
      const result = await retrievalChain.invoke({ input: question, chat_history: chatHistory.chat_history });

      await memory.saveContext({ question: question }, { answer: result.answer });
      console.log("[INFO] Successfully answered from documents. Sending response.");
      return res.status(200).json({ answer: result.answer });
    } else {
      // Fallback to web search
      console.log("[INFO] No relevant documents found. Falling back to web search.");
      const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_SEARCH_API_KEY}&cx=${GOOGLE_SEARCH_ENGINE_ID}&q=${encodeURIComponent(question)}`;
      const searchResponse = await axios.get(searchUrl);
      const searchResults = searchResponse.data.items || [];
      let searchContext = '';
      searchResults.forEach(item => { searchContext += `Title: ${item.title}\nLink: ${item.link}\nSnippet: ${item.snippet}\n\n`; });

      if (searchContext) {
        console.log(`[DEBUG] Built search context of ${searchContext.length} chars.`);
        const finalPrompt = `You are an AI assistant that answers questions based on the following internet search results. Answer as accurately and concisely as possible. Search Results: ${searchContext}\n\nQuestion: ${question}`;
        console.log(`[DEBUG] Final prompt for model (first 150 chars): ${finalPrompt.substring(0, 150)}`);
        const result = await model.invoke(finalPrompt);

        console.log("[INFO] Successfully answered from web search fallback. Sending response.");
        return res.status(200).json({ answer: result.content });
      } else {
        console.log("[INFO] Web search fallback yielded no results. Sending final message.");
        return res.status(200).json({ answer: "I could not find a relevant answer in your documents or on the internet." });
      }
    }
  } catch (error) {
    console.error("[FATAL] Error in handler:", error);
    res.status(500).json({ error: error.message || "An error occurred." });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
