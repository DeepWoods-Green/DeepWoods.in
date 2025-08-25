// index.js
import express from "express";
import dotenv from "dotenv";
import chatHandler from "./api/chat.js";

dotenv.config();

const app = express();
app.use(express.json());

// Route for your PDF chatbot
app.post("/api/chat", chatHandler);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
