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


// Global constants
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const GOOGLE_SEARCH_API_KEY = process.env.GOOGLE_SEARCH_API_KEY;
const GOOGLE_SEARCH_ENGINE_ID = process.env.GOOGLE_SEARCH_ENGINE_ID;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;


// Simple in-memory store for conversation memory.
const memory = new BufferMemory({
    memoryKey: "chat_history",
    inputKey: "question",
    outputKey: "answer",
    returnMessages: true
});


// New welcome route to handle browser requests
app.get('/', (req, res) => {
    res.send('Welcome to the Deepwoods AI Assistant!');
});


// Your API endpoint for the chatbot
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


        // The corrected logic for your smarter chatbot
        if (pdfUrl && pdfUrl !== 'general_discussion') {
            // Logic for when a specific PDF is selected
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

            let retrievedDocs;
            try {
                retrievedDocs = await retriever.getRelevantDocuments(question);
            } catch (error) {
                console.error("Error fetching vector documents:", error);
                return res.status(500).json({ error: "Vector search error" });
            }


            // Check if relevant documents were found
            if (retrievedDocs && retrievedDocs.length > 0) {
               const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Answer the user's question using the context and history.\n<chat_history>{chat_history}</chat_history>\n<context>{context}</context>"],
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


                const chatHistory = await memory.loadMemoryVariables({});
                const result = await retrievalChain.invoke({
                    input: question,
                    chat_history: chatHistory.chat_history
                });


                await memory.saveContext({ question: question }, { answer: result.answer });
                return res.status(200).json({ answer: result.answer });
            } else {
                // Fallback to web search ONLY if a PDF was selected but no relevant docs were found
                const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_SEARCH_API_KEY}&cx=${GOOGLE_SEARCH_ENGINE_ID}&q=${encodeURIComponent(question)}`;
                
                const searchResponse = await axios.get(searchUrl);
                const searchResults = searchResponse.data.items || [];
                
                let searchContext = '';
                searchResults.forEach(item => {
                    searchContext += `Title: ${item.title}\nLink: ${item.link}\nSnippet: ${item.snippet}\n\n`;
                });


                if (searchContext) {
                    const prompt = ChatPromptTemplate.fromMessages([
                        ["system", "You are an AI assistant that answers questions based on the following internet search results. Answer as accurately and concisely as possible. Search Results: {context}\n\nQuestion: {question}"]
                    ]);
                    
                    const result = await model.invoke(prompt.format({
                        context: searchContext,
                        question: question
                    }));


                    const warning = "⚠️ Note: I couldn't find an answer in your documents, so the response below was generated based on an internet search.";
                    return res.status(200).json({ answer: `${warning}\n\n${result.content}` });
                } else {
                    return res.status(200).json({ answer: "I could not find a relevant answer in your documents or on the internet." });
                }
            }
        } else {
            // General discussion mode (no PDF selected)
            const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_SEARCH_API_KEY}&cx=${GOOGLE_SEARCH_ENGINE_ID}&q=${encodeURIComponent(question)}`;
            
            const searchResponse = await axios.get(searchUrl);
            const searchResults = searchResponse.data.items || [];
            
            let searchContext = '';
            searchResults.forEach(item => {
                searchContext += `Title: ${item.title}\nLink: ${item.link}\nSnippet: ${item.snippet}\n\n`;
            });
            
            if (searchContext) {
                const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are an AI assistant that answers questions based on the following internet search results. Answer as accurately and concisely as possible. Search Results: {context}\n\nQuestion: {question}"]
]);

// Format prompt to ChatPromptValue object (not string)
const promptValue = await prompt.formatPromptValue({
  context: searchContext,
  question: question
});

// Optional: debug the messages sent to the model
console.log(promptValue.toChatMessages());

// Call model.invoke() with the promptValue object
const result = await model.invoke(promptValue);


// Pass the prompt value to model.invoke() which accepts prompt objects
const result = await model.invoke(promptValue);





                return res.status(200).json({ answer: result.content });
            } else {
                return res.status(200).json({ answer: "I could not find a relevant answer on the internet." });
            }
        }
    } catch (error) {
        console.error("Error in handler:", error);
        res.status(500).json({ error: "An error occurred." });
    }
});


app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
