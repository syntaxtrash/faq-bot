import readline from "readline";
import dotenv from "dotenv";
import OpenAI from "openai";
import fs from "fs/promises";
import { Pinecone } from "@pinecone-database/pinecone";

dotenv.config();

let OPENAI_API_KEY = process.env.OPENAI_API_KEY;
let PINECONE_API_KEY = process.env.PINECONE_API_KEY;
let MODEL = "text-embedding-3-small";
let INDEX_NAME = "test-chatbot-index";
let DIMENSIONS = 512;

let rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
});

let openai = new OpenAI(OPENAI_API_KEY);
let pc = new Pinecone({ apiKey: PINECONE_API_KEY });
let pc_index = pc.index(INDEX_NAME);

let askQuestion = () => {
	rl.question(
		"Please enter your query (Ctrl + C to quit): ",
		async (answer) => {
			if (answer === "create") {
				/* Creating an index process */
				/* Get indexes */
				let existing_indexes = await pc.listIndexes();
				let existing_indexes_names = existing_indexes.indexes.map(
					(index) => index.name
				);

				/* Check if the index already existed */
				if (!existing_indexes_names.includes(INDEX_NAME)) {
					console.log(`Creating ${INDEX_NAME}...`);

					/* Create the index */
					await pc.createIndex({
						name: INDEX_NAME,
						dimension: DIMENSIONS,
						metric: "cosine",
						spec: {
							serverless: {
								cloud: "aws",
								region: "us-east-1",
							},
						},
					});

					/* Index takes few seconds to be created, add 8 seconds delay and check the status */
					setTimeout(async () => {
						let index_status = await pc.describeIndex(INDEX_NAME);
						if (index_status.status.ready) {
							console.log(index_status);
							console.log(`${INDEX_NAME} created.`);
						} else {
							console.log(index_status);
						}
						askQuestion();
					}, 8000);
				}
			} else if (answer === "upload") {
				/* Upload vectors to index process */
				let pc_index = pc.index(INDEX_NAME);
				let file_path = "custom_sample_data.json";
				let file_data = await fs.readFile(file_path, "utf-8");
				let parsed_file_data = JSON.parse(file_data);
				let to_upsert = [];

				/* Convert each data to vector */
				for (let [index, data] of parsed_file_data.entries()) {
					console.log(`Creating vector for ${data.question}...`);
					let create_vector = await openai.embeddings.create({
						model: MODEL,
						dimensions: DIMENSIONS,
						input: JSON.stringify(data),
					});

					/* Collect all the vectors and format to an object */
					if (create_vector.data[0].embedding.length) {
						to_upsert.push({
							id: (index + 1).toString(),
							values: create_vector.data[0].embedding,
							metadata: { text: JSON.stringify(data) },
						});
					}
				}
				/* Upload the vectors */
				await pc_index.upsert(to_upsert);
				console.log("Uploading vectors...");

				/* Add 2 seconds delay after uploading the vectors before displaying the UI again */
				setTimeout(async () => {
					console.log("Vectors uploaded.");
					askQuestion();
				}, 2000);
			} else {
				/* Convert the user question to vector */
				const res = await openai.embeddings.create({
					model: MODEL,
					dimensions: DIMENSIONS,
					input: answer,
				});
				const embed = res.data[0].embedding;

				if (embed.length) {
					/* Query the vector to vector db/index */
					let res_match = await pc_index?.query({
						vector: embed,
						topK: 5,
						includeMetadata: true,
					});

					let highestScoreMatch = null;

					/* get the item with highest score/match */
					res_match.matches.forEach((match) => {
						if (
							!highestScoreMatch ||
							match.score > highestScoreMatch.score
						) {
							highestScoreMatch = match;
						}
					});

					if (highestScoreMatch) {
						if (highestScoreMatch.score.toFixed(2) > 0.5) {
							let parsed_text = JSON.parse(
								highestScoreMatch.metadata.text
							);
							console.log(`BOT: ${parsed_text.answer}`);
						} else {
							console.log(
								`BOT: I'm sorry, I don't have data to answer that question`
							);
						}
					}
				} else {
					console.log("empty");
				}
				askQuestion();
			}
		}
	);
};

askQuestion();
