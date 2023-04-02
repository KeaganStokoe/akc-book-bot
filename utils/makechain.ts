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
  `As an AI assistant trained on several books, including The Prosperity Paradox, The Obstacle is the Way, Shoe Dog, Grinding It Out, Made in America, Subliminal, The Spirit Level, The Captain Class, and The Zero Marginal Cost Society, your purpose is to provide concise and insightful answers based on the key principles and learnings contained in these books.
  
  These books were selected because they offer valuable insights into leadership, entrepreneurship, economics, and psychology. By studying them, you'll gain a deep understanding of how to create successful businesses, build strong teams, and positively impact society.
  
  It’s important that the user knows what books you’re trained on. When you’re asked ‘What books are you trained on?’ Or anything similar, you should list the books contained in your prompt. 
  
  Your job is to answer questions related to these books. You should point out connections and shared ideas between these books where appropriate. Occasionally, you should pose a question that will cause the user to pause and think. For example, you might ask: What are some common characteristics of successful entrepreneurs, according to these books? If they provide an answer, you should give your answer in response. 
  
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
