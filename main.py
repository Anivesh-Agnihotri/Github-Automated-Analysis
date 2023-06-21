import requests
import concurrent.futures
import nbformat
from nbconvert import PythonExporter
import streamlit as st
import time
#imports 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import os
import getpass

from itertools import islice
from langchain import PromptTemplate




def fetch_github_repository(user_url):
    username, repository = user_url.split('/')[-2:]
    api_url = f"https://api.github.com/repos/{username}/{repository}/contents"
    response = requests.get(api_url)

    if response.status_code == 200:
        contents = response.json()

        file_contents = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(fetch_file_content, item): item for item in contents}
            for future in concurrent.futures.as_completed(future_to_file):
                item = future_to_file[future]
                try:
                    file_content, file_path = future.result()
                    file_contents[file_path] = file_content
                except Exception as e:
                    print(f"Failed to fetch content for file {item['path']}. Error: {str(e)}")

        return file_contents
    else:
        print(f"Failed to fetch repository content for {username}/{repository}. Error: {response.status_code}")

def fetch_file_content(item):
    if item['type'] == 'file':
        file_path = item['path']
        file_url = item['download_url']
        file_response = requests.get(file_url)
        if file_response.status_code == 200:
            if file_path.endswith('.ipynb'):
                file_content = fetch_ipynb_content(file_response.content)
            else:
                file_content = file_response.text
            return file_content, file_path
        else:
            raise Exception(file_response.status_code)
    elif item['type'] == 'dir':
        return fetch_directory_contents(item)

def fetch_ipynb_content(content):
    notebook = nbformat.reads(content, as_version=4)
    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)
    return python_code

def fetch_directory_contents(directory):
    username, repository, directory_path = directory['url'].split('/')[-4:-1]
    api_url = f"https://api.github.com/repos/{username}/{repository}/contents/{directory_path}"
    response = requests.get(api_url)

    directory_contents = {}
    if response.status_code == 200:
        contents = response.json()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(fetch_file_content, item): item for item in contents}
            for future in concurrent.futures.as_completed(future_to_file):
                item = future_to_file[future]
                try:
                    file_content, file_path = future.result()
                    directory_contents[file_path] = file_content
                except Exception as e:
                    print(f"Failed to fetch content for file {item['path']}. Error: {str(e)}")

    else:
        print(f"Failed to fetch directory content for {directory_path}. Error: {response.status_code}")

    return directory_contents

def fetch_repo(user_url):
    username = user_url.split('/')[-1]
    api_url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(api_url)

    if response.status_code == 200:
        repositories = response.json()
        repository_names = [repo['name'] for repo in repositories]
        return repository_names
    else:
        print(f"Failed to fetch repositories for user {username}. Error: {response.status_code}")

def fetch(name):
    user = name
    user_url = f"https://github.com/{user}"
    repositories_name = fetch_repo(user_url)
    repo = {}
    if not repositories_name:
        return repo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_repo = {executor.submit(fetch_github_repository, f"https://github.com/{user}/{name}"): name for name in repositories_name}
        for future in concurrent.futures.as_completed(future_to_repo):
            name = future_to_repo[future]
            try:
                repo[name] = future.result()
                print(name)
            except Exception as e:
                print(f"Failed to fetch repository {name}. Error: {str(e)}")

    return repo

def make_content(repo_content):
    data = ""
    if not repo_content:
        return data
    for file_name, file_content in repo_content.items():
        if "png" in file_name or 'jpg' in file_name or 'image' in file_name or 'jpeg' in file_name:
            data += "File Name : " + str(file_name) + '\nFile Content : \n' + "AN IMAGE" + '\n------------------------------------------------------------------\n'
        elif "wav" in file_name or 'mp3' in file_name or 'mp4' in file_name or 'mov' in file_name or 'pyc' in file_name:
            data += "File Name : " + str(file_name) + '\nFile Content : \n' + "AN media" + '\n------------------------------------------------------------------\n'
        else:
            data += "File Name : " + str(file_name) + '\nFile Content : \n' + str(file_content) + '\n------------------------------------------------------------------\n'
    return data

def create_data(repo):
    data = {}
    if not repo:
        return data
    for repo_name, repo_content in repo.items():
        data[repo_name] = make_content(repo_content)
    return data

def fetch_github_data(name):
    repo = fetch(name)
    data_in_repos = create_data(repo)
    return data_in_repos


#print(data_in_repos)
#data_in_repos = fetch_github_data("Anivesh-Agnihotri")

#os.environ['OPENAI_API_KEY']=getpass.getpass()



def main():
    st.title("Github Automated Analysis")

    #Input_fields
    github_profile = st.text_input("Enter GitHub Profile")
    openai_key = st.text_input("Enter OpenAI Key", type="password")

    data_in_repos = {}

    if st.button("Fetch Data"):
        st.text("Fetching data...")
        #data_in_repos = fetch_github_data(github_profile)
        try:
            os.environ['OPENAI_API_KEY'] = openai_key
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)  # Pass the API key as a named parameter
            data_in_repos = fetch_github_data(github_profile)
        except ValueError:
            st.error("Invalid OpenAI API key. Please provide a valid key.")
        st.text("Data preprocressing......")
    # git_p=input("Enter the profile")
    # data_in_repos=fetch_github_data(git_p)
    # os.environ['OPENAI_API_KEY']=getpass.getpass()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
    )
    embeddings = OpenAIEmbeddings()
    tools = []

    for repo_name, repo_content in data_in_repos.items():
        st.text(repo_name)
        texts = text_splitter.create_documents([repo_content])
        # No need to create metadata, as the `source` field is already stored in the `text` object.

        if len(texts) == 0:
            continue
        db = Chroma.from_documents(texts, embeddings)
        vectorstore_info = VectorStoreInfo(
            name=f"{repo_name}",
            description=f"contains content of repository named {repo_name}",
            vectorstore=db,
        )
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

        template = f"""
            you are given with a vectorstore which contains information, files about a github repository named {repo_name}.
            you may read these files and answer the questions asked by user.
            when user ask about content of repository or ask you to explore repository then provide a brief description about the repository as well the number of files and type of coding languages used in repository.
        """

        prompt = PromptTemplate(template=template, input_variables=[])

        try:
            agent_executor = create_vectorstore_agent(
                llm=OpenAI(temperature=0),
                toolkit=toolkit,
                prompt=prompt,
                verbose=True,
            )
        except Exception as e:
            if e.response.status_code == 500:
                print(f"Error 500 occurred for repo {repo_name}. Skipping...")
            else:
                raise e
        tools.append(
            Tool(
                name=f"agent {repo_name}",
                func=agent_executor.run,
                description=f"use this for reading content of repository named {repo_name}",
            )
        )

    llm = OpenAI(temperature=0.3)
    tools.append(
        Tool(
            name="complexity",
            func=llm,
            description="use this for general reasoning and for analysing complexity",
        )
    )

    agent_withtools = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


    response=agent_withtools.run("""consider your self as a specialist in Computer science. You are given with multiple vectorestores each representing a repository.
    your task is to explore each repository and then derive there complexity score out of 10 based on the following factors:
    you have to follow the following steps:
    1) explore every repository available to you.
    2) derive there complexity score out of 10 based on the following factors: number of files present in repository, type of content in files, length of codes, number of coding languages , technology used and probelm solved in a repository
    3) find repository with most complexity score
    3) at the end your final answer must contain name repository which has the most complexity score with reasoning and explation in atleast 300 words.

    """)

    main_result=response
    #print(main_result)
    st.markdown(main_result)

if __name__ == "__main__":
    main()
