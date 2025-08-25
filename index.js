// index.js
import express from "express";
import dotenv from "dotenv";
import chatHandler from "./api/chat.js";

dotenv.config();

const app = express();
app.use(express.json());

// PDF chatbot route
app.post("/api/chat", chatHandler);

// Root route for health check
app.get("/", (req, res) => {
  res.send("PDF Chatbot backend is running.");
});

// Use PORT from Render environment
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
