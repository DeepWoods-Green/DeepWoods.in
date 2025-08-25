// api/chat.js
import fetch from "node-fetch";
import pdf from "pdf-parse";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core";
import { RetrievalQAChain } from "@langchain/core/dist/chains";
import { RecursiveCharacterTextSplitter } from "@langchain/core/dist/text_splitter";
import { MemoryVectorStore } from "@langchain/core/dist/vectorstores/memory";

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { question, pdfUrl } = req.body;

    if (!question || !pdfUrl) {
      return res.status(400).json({ error: "Missing question or pdfUrl" });
    }

    // Fetch PDF
    const response = await fetch(pdfUrl);
    if (!response.ok) throw new Error("Failed to fetch PDF");

    const arrayBuffer = await response.arrayBuffer();
    const pdfBuffer = Buffer.from(arrayBuffer);

    // Parse PDF text
    const data = await pdf(pdfBuffer);
    const rawText = data.text;

    // Split text into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splits = await textSplitter.createDocuments([rawText]);

    // Create embeddings and vector store
    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "embedding-001",
      apiKey: GOOGLE_API_KEY,
    });
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

    // Setup the LLM
    const model = new ChatGoogleGenerativeAI({
      modelName: "gemini-1.5-flash-latest",
      apiKey: GOOGLE_API_KEY,
    });

    // Define prompt template
    const prompt = PromptTemplate.fromTemplate(
      "You are an AI assistant that provides accurate information from the following context. " +
      "If the answer is not in the context, say 'I cannot answer that question based on the provided documents.' " +
      "Do not make up an answer. Context: {context}\n\nQuestion: {question}"
    );

    // Create retrieval chain
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), { prompt });

    // Ask the question
    const result = await chain.invoke({ query: question });
    const answer = result?.text || result?.output_text || result || "No answer returned";

    res.status(200).json({ answer });
  } catch (error) {
    console.error("Error in PDF chat handler:", error);
    res.status(500).json({ error: error.message || "An error occurred." });
  }
}
