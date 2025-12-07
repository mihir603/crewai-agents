import agentops
from textwrap import dedent
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(dotenv_path="./OAI_CONFIG_LIST")
agentops.init()

print("Building a JS Game")
print("-------------------------------")
game = input("What game do you want to build ? What will be the mechanics ?")
coding_agent = Agent(
  role = "Senior Software Developer",
  goal = "Create the required software as needed",
  backstory = dedent(
    """ 
      You are a Senior Fullstack Web Developer at leading think tank.
      Your expertise in programming in JavaScript. and do your best 
      to produce perfect code.
    """
  ),
  allow_delegation = False,
  verbose = True
)

testing_agent = Agent(
  role = "Quality Assurance Engineer",
  goal = "create perfect code, by analyzing the given code for errors",
  backstory = dedent(
    """
      You are a software engineer that specializes in checking code
      for errors. You have an eye for detail and a knack for finding
      hidden bugs.
      You check for missing imports, variable declarations, mismatched
      brackets and syntax errors.
      You also check for security vulnerabilities, and logic errors.
    """
  ),
  allow_delegation = False,
  verbose = True
)

eng_manager_agent = Agent(
  role = "Chief Software and Quality Control Engineer",
  goal = "Ensure that code does the job that its supposed to do.",
  backstory = dedent(
    """ 
    You are a Chief Software Quality Control Engineer at a leading
        tech think tank. You are responsible for ensuring that the code
        that is written does the job that it is supposed to do.
        You are responsible for checking the code for errors and ensuring
        that it is of the highest quality.

    """
  ),
  allow_delegation = False,
  verbose = True
)

code_task = Task(
  description=f"""
    You will create the game using JavaScript, these are the instructions:
    Instructions:
    -------------
    {game}
    You will write the code for the game using JavaScript.
 """,
 expected_output="Your Final Answer must be full JavaScript Code, only HTML, CSS and JS code and nothing else.",
 agent=coding_agent
)

test_task = Task(
  description=f"""
      You are helping creating a game using JavaScript, here are the instructions:
      Instructions:
      --------------
      {game}
      Using the code you got, check for errors, logical errors, import errors,
      syntax errors, bracket mismatch errors and security vulnerabilities.
     """,
     expected_output="Output the list of issues you found in the code",
     agent = testing_agent
)


evaluate_task = Task(
  description=f""" 
      You are helping creating a game using JavaScript, here are the instructions:
      Instructions:
      -------------
      {game}
      You will look over the code and insure that it is complete and does the job
      that its supposed to do.
      """,
      expected_output="Your final output must be the corrected full JavaScript code, just HTML,CSS, JavaScript code and nothing else.",
      agent=eng_manager_agent
)


crew = Crew(
  agents=[coding_agent, 
          testing_agent, 
          eng_manager_agent],
  tasks=[code_task, 
         test_task, 
         evaluate_task],
  verbose=True,
  process=Process.hierarchical,
  manager_llm= ChatOpenAI(
    temperature=0,
    model="gpt-5"
  )
)

result = crew.kickoff()
print("###############################")
print(result)