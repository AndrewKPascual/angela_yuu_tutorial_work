import dotenv from 'dotenv';
dotenv.config();

import express from "express";
import bodyParser from "body-parser";
import path from 'path';
import { fileURLToPath } from 'url';
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import pg from 'pg';
const { Pool } = pg;

const pool = new Pool({
  host: process.env.DB_HOST,
  port: 5432,
  user: 'postgres',
  password: process.env.DB_PASSWORD,
  database: 'postgres',
});

const __dirname = path.dirname(fileURLToPath(import.meta.url));


const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

const port = 3000;
const apiKey = process.env.OPENAI_API_KEY;

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'project.html'));
});

app.use(express.static(path.join(__dirname, 'public')));

const model = new ChatOpenAI({});
const vectorStore = await HNSWLib.fromTexts(
  ["mitochondria is the powerhouse of the cell"],
  [{ id: 1 }],
  new OpenAIEmbeddings()
);
const retriever = vectorStore.asRetriever();

const prompt = PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

const chain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocumentsAsString),
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
]);

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'project.html'));
});

app.post("/ask", async (req, res) => {
  const question = req.body.question;
  const result = await chain.invoke(question);

  // Save to database
  const query = {
    text: 'INSERT INTO public.questions_responses(question, response) VALUES($1, $2)',
    values: [question, result],
  };

  try {
    await pool.query(query);
  } catch (err) {
    console.error(err);
  }

  res.send(result);
});

app.listen(port, () => {
  console.log(`Listening on port ${port}`);
});
