import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { createClient } from "@supabase/supabase-js";
import { v4 as uuidv4 } from 'uuid';
import * as PDFJS from 'pdfjs-dist/legacy/build/pdf.js';
import dotenv from 'dotenv';
dotenv.config();

const { getDocument } = PDFJS.default || PDFJS;

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

const pdfUrls = [
    "https://greenpositive.org/wp-content/uploads/2025/08/Featherlite-Furniture-FL-GHG-Emissions-Report-FY25-V.2.0.pdf",
    "https://greenpositive.org/wp-content/uploads/2025/08/Featherlite-GHG-Emissions-Inventory-Report-2024-Final-Version.pdf",
    "https://greenpositive.org/wp-content/uploads/2025/08/Featherlite_FL_GHG-Emissions-Report-_FY23-Final-Version.pdf",
    "https://greenpositive.org/wp-content/uploads/2025/08/Featherlite-Sustainability-Report-2025.pdf",
    // This PDF appears to be corrupt, so we've commented it out.
    // "https://greenpositive.org/wp-content/uploads/2025/08/Featherlite-Sustainability-PPT_August-2025_Ver-2.pdf",
    "https://greenpositive.org/wp-content/uploads/2025/08/Featherlite-SDGs-Alignment-Sustainability-Report-FY25.pdf",
    "https://greenpositive.org/wp-content/uploads/2025/08/Sustainability-Communication-Deepwoods-Green-Profile-2025.pdf",
    "https://greenpositive.org/wp-content/uploads/2025/08/Deepwoods-Green-Service-Ppt-August-2025_Ver-2.pdf"
];

async function ingestDocuments() {
    try {
        const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
        const embeddings = new GoogleGenerativeAIEmbeddings({
            modelName: "embedding-001",
            apiKey: GOOGLE_API_KEY,
        });

        for (const pdfUrl of pdfUrls) {
            console.log(`Processing PDF from URL: ${pdfUrl}`);
            const pdfResponse = await fetch(pdfUrl);
            const arrayBuffer = await pdfResponse.arrayBuffer();
            const pdfData = new Uint8Array(arrayBuffer);

            const pdfDocument = await getDocument({ data: pdfData }).promise;
            let rawText = '';
            for (let i = 1; i <= pdfDocument.numPages; i++) {
                const page = await pdfDocument.getPage(i);
                const content = await page.getTextContent();
                rawText += content.items.map(item => item.str).join(' ') + ' ';
            }
            
            // Clean the text to remove null characters that cause the database error
            const cleanText = rawText.replace(/\\u0000/g, '');

            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200
            });
            const splits = await textSplitter.createDocuments([cleanText]);

            for (const split of splits) {
                const embedding = await embeddings.embedDocuments([split.pageContent]);
                const { data, error } = await supabase
                    .from('documents')
                    .insert([
                        {
                            id: uuidv4(),
                            content: split.pageContent,
                            embedding: embedding[0],
                        }
                    ]);

                if (error) {
                    console.error("Error inserting data:", error);
                } else {
                    console.log(`Successfully ingested a document chunk from ${pdfUrl}`);
                }
            }
        }
        console.log("Document ingestion complete.");
    } catch (error) {
        console.error("An error occurred during ingestion:", error);
    }
}

ingestDocuments();