from LLMs.falconsai import FalconSai
from LLMs.bart_large import BartLarge
from LLMs.led_large_book import LEDLargeBook
from LLMs.falcon_7b_instruct import Falcon7b
from LLMs.vicuna_7b import Vicuna7b
from LLMs.llama2_quant import Llamma7b
import os

def getText(folder_path):
    result_dict = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                # Create a dictionary from keys and values
                result_dict.update({filename: file_content})

    return result_dict

'''
FalconSai()
BartLarge()
LEDLargeBook()
Falcon7b()
Vicuna7b()
'''

model = LEDLargeBook()

modelName = model.getName()
textFolder = "./BookText/"
resultsFolder = "./testResults"
promptsFolder = "./Prompts/"
promptsDict = getText(textFolder)
promptsInit = {"":""}

testNumber = 1

print(f'testing {modelName}')
for filename, text in promptsDict.items():
    print(f'\nstarting test {testNumber} using {filename}')
    if (model.usesPrompts):
        promptsInit = getText(promptsFolder)

    promptNum = 1
    for promptFile, promptText in promptsInit.items():
        fullPrompt = promptText + text
        results = model.run(fullPrompt)
        print(f'\nRESULTS:\n{results}\n')
        
        test_file_name = modelName + '_' + filename.replace('.txt','') + '_' + promptFile.replace('.txt','') + ".txt"

        # Combine the folder path and file name
        file_path = os.path.join(resultsFolder , test_file_name)
        # Create the text file and write content
        with open(file_path, 'w') as file:
            file.write(results)
        print(f'Finished writing results to {test_file_name}')
        promptNum += 1

    testNumber += 1


 