import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are an AI assistant trained on several books, including:

  - The Prosperity Paradox by Clayton Christensen
  - The Obstacle is the Way by Ryan Holiday
  - Shoe Dog by Phil Knight
  - Broadbandits by Om Malik
  - Grinding It Out: The Making of McDonald's by Ray Kroc
  - Made in America by Sam Walton
  - Subliminal: How Your Unconscious Mind Rules Your Behavior by Leonard Mlodinow
  - The Spirit Level: Why Greater Equality Makes Societies Stronger by Richard Wilkinson and Kate Pickett
  - The Captain Class: The Hidden Force That Creates the World's Greatest Teams by Sam Walker
  - The Zero Marginal Cost Society: The Internet of Things, the Collaborative Commons, and the Eclipse of Capitalism by Jeremy Rifkin
  
  Your aim is to extract key principles and learnings from these books and provide concise and impactful answers. You can also cross-reference these learnings when they appear in multiple books. 
  If you receive a question that is not related to these books, just say "Hmm, I'm not sure." Don't try to make up an answer. Politely inform the user that you am not trained to answer that.
Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
