import express from 'express';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RetrievalQAChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import * as PDFJS from 'pdfjs-dist/build/pdf.node.mjs';

const app = express();
const port = process.env.PORT || 3000;
app.use(express.json());

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

// Your API endpoint for the chatbot
app.post('/api/chat', async (req, res) => {
    try {
        const { question, pdfUrl } = req.body;

        if (!question || !pdfUrl) {
            return res.status(400).json({ error: "Missing question or pdfUrl" });
        }

        // Fetch the PDF data
        const pdfResponse = await fetch(pdfUrl);
        const arrayBuffer = await pdfResponse.arrayBuffer();
        const pdfData = new Uint8Array(arrayBuffer);

        // Parse the PDF text using pdfjs-dist
        const pdfDocument = await PDFJS.getDocument({ data: pdfData }).promise;
        let rawText = '';
        for (let i = 1; i <= pdfDocument.numPages; i++) {
            const page = await pdfDocument.getPage(i);
            const content = await page.getTextContent();
            rawText += content.items.map(item => item.str).join(' ') + ' ';
        }

        // Split the documents into smaller chunks
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
        const splits = await textSplitter.createDocuments([rawText]);

        // Create embeddings and a vector store
        const embeddings = new GoogleGenerativeAIEmbeddings({
            modelName: "embedding-001",
            apiKey: GOOGLE_API_KEY,
        });
        const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

        // Create a retrieval chain to answer questions
        const model = new ChatGoogleGenerativeAI({
            modelName: "gemini-1.5-flash-latest",
            apiKey: GOOGLE_API_KEY,
        });

        const prompt = PromptTemplate.fromTemplate(
            "You are an AI assistant that provides accurate information from the following context. If the answer is not in the context, say 'I cannot answer that question based on the provided documents.' Do not make up an answer. Context: {context}\\n\\nQuestion: {question}"
        );

        const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
            prompt: prompt,
        });

        const result = await chain.invoke({
            query: question
        });

        res.status(200).json({ answer: result.text });
    } catch (error) {
        console.error("Error in handler:", error);
        res.status(500).json({ error: "An error occurred." });
    }
});

// Listen for incoming requests
app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});