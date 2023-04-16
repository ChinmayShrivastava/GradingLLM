import openai
import os
import PyPDF2
import numpy as np
import pandas as pd

openai.api_key = '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'

# function to get the embedding of a text using GPT-3
def gpt3_embedding(content, engine='text-embedding-ada-002'):
	response = openai.Embedding.create(input=content,engine=engine)
	vector = response['data'][0]['embedding']  # this is a normal list
	return vector

# function to return cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))

def gpt3_3turbo_completion(messages, summarymodel='gpt-4'): # 'gpt-3.5-turbo' or 'gtp-4'

	temperature = 0.2
	max_length = 150
	top_p = 1.0
	frequency_penalty = 0.0
	presence_penalty = 0.1

	# making API request and error checking
	print("making API request")
	try:
		response = openai.ChatCompletion.create(
			model=summarymodel, 
			messages=messages, 
			temperature=temperature, 
			max_tokens=max_length, 
			top_p=top_p, 
			frequency_penalty=frequency_penalty, 
			presence_penalty=presence_penalty)
	except openai.error.RateLimitError as e:
		print("OpenAI API rate limit error! See below:")
		print(e)
		return None, None, None
	except Exception as e:
		print("Unknown OpenAI API error! See below:")
		print(e)
		return None, None, None
	
	return response['choices'][0]['message']['content']
     
if __name__ == '__main__':

    # # read the rubric file as a text file, and break each line and make a list of lines
    # with open('rubric.txt', 'r') as f:
    #     rubric = f.read().splitlines()
    
    # [print(x, '\n') for x in rubric]
    # wait = input("PRESS ENTER TO CONTINUE.")

    # # read teh solution set file as a text file, and break each line and make a list of lines
    # with open('solutions.txt', 'r') as f:
    #     solutions = f.read().splitlines()

    # [print(x, '\n') for x in solutions]
    # wait = input("PRESS ENTER TO CONTINUE.")

    # # get the embedding of each line of the rubric
    # rubricembeddings = []
    # rubricchunks = []
    # for line in rubric:
    #     rubricembeddings.append(gpt3_embedding(line))
    #     rubricchunks.append(line)

    # # get the embedding of each line of the solution set
    # solutionembeddings = []
    # solutionchunks = []
    # for line in solutions:
    #     question = line.split('::')[0]
    #     solution = line.split('::')[1]
    #     prompt = question + '\n' + solution
    #     # get the embeddings of each 256 character chunk of the solution
    #     for i in range(0, len(prompt), 256):
    #         if i + 256 > len(prompt):
    #             solutionembeddings.append(gpt3_embedding(prompt[i:]))
    #             solutionchunks.append(prompt[i:])
    #         else:
    #             solutionembeddings.append(gpt3_embedding(prompt[i:i+256]))
    #             solutionchunks.append(prompt[i:i+256])
    
    # # save all the four lists
    # np.save('rubricembeddings.npy', rubricembeddings)
    # np.save('rubricchunks.npy', rubricchunks)
    # np.save('solutionembeddings.npy', solutionembeddings)
    # np.save('solutionchunks.npy', solutionchunks)

#=============================================================================================================
# if the embeddings are already saved, then load them, comment the above code and uncomment the below code
#=============================================================================================================

    # load the four files
    rubricembeddings = np.load('rubricembeddings.npy')
    print(rubricembeddings[0:5])
    wait = input("PRESS ENTER TO CONTINUE.")
    rubricchunks = np.load('rubricchunks.npy')
    print(rubricchunks[0:5])
    wait = input("PRESS ENTER TO CONTINUE.")
    solutionembeddings = np.load('solutionembeddings.npy')
    print(solutionembeddings[0:5])
    wait = input("PRESS ENTER TO CONTINUE.")
    solutionchunks = np.load('solutionchunks.npy')
    print(solutionchunks[0:5])
    wait = input("PRESS ENTER TO CONTINUE.")

    # iterate through the submissions folder and read one pdf at a time
    for filename in os.listdir('submissions'):
         if filename.endswith('.pdf'):

            print(filename)
            pdfFileObj = open('submissions/' + filename, 'rb')
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            num_pages = pdfReader.numPages
            count = 0
            text = ""

            # The while loop will read each page
            while count < num_pages:
                pageObj = pdfReader.getPage(count)
                count +=1
                text += pageObj.extractText()

            tempvlist = []
            tempchunks = []
            # get the embedding of each 256 character chunk of the submission
            for i in range(0, len(text), 256):
                if i + 256 > len(text):
                    tempvlist.append(gpt3_embedding(text[i:]))
                    tempchunks.append(text[i:])
                else:
                    tempvlist.append(gpt3_embedding(text[i:i+256]))
                    tempchunks.append(text[i:i+256])

            studentdata = []
                
            # iterate through each rubric item
            for i in range(len(rubricembeddings)):

                # find the top ten similar vectors from the solution as well as the submission
                rubricvector = rubricembeddings[i]

                # find the top ten similar vectors from the solution
                solutionvectors = []
                solutionchunksfound = []
                for j in range(len(solutionembeddings)):
                    solutionvectors.append(cosine_similarity(rubricvector, solutionembeddings[j]))
                solutionvectors = np.array(solutionvectors)
                solutionvectors = solutionvectors.argsort()[-10:][::-1]
                for j in range(len(solutionvectors)):
                    solutionchunksfound.append(solutionchunks[solutionvectors[j]])

                # find the top ten similar vectors from the submission
                submissionvectors = []
                submissionchunksfound = []
                for j in range(len(tempvlist)):
                    submissionvectors.append(cosine_similarity(rubricvector, tempvlist[j]))
                submissionvectors = np.array(submissionvectors)
                submissionvectors = submissionvectors.argsort()[-10:][::-1]
                for j in range(len(submissionvectors)):
                    submissionchunksfound.append(tempchunks[submissionvectors[j]])

                # prepare the prompt for GPT-3
                prompt = 'Rubric Item: ' + rubricchunks[i] + '\n\n' + 'Solution Chunks: ' + '\n\n'
                for j in range(len(solutionchunksfound)):
                    prompt += str(j+1) + '. ' + solutionchunksfound[j] + '\n\n'
                prompt += 'Submission Chunks: ' + '\n\n'
                for j in range(len(submissionchunksfound)):
                    prompt += str(j+1) + '. ' + submissionchunksfound[j] + '\n\n'
                
                # get the gpt-4 completion for the prompt
                messages = [{'role': 'user', 'content': prompt}, {'role': 'user', 'content': 'Return a rating for the rubric item and an extremely short comment for the student. Be generous.'}]
                response = gpt3_3turbo_completion(messages, summarymodel='gpt-4')

                # append the response to the studentdata list
                studentdata.append(response)
                print(response)

            # save the studentdata list as a csv file
            df = pd.DataFrame(studentdata)
            df.to_csv('results/' + filename.split('.')[0] + '.csv', index=False)

            # close the pdf file
            pdfFileObj.close()