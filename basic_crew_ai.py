from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv(dotenv_path="./OAI_CONFIG_LIST")

# creating a simple tech researcher
tech_researcher = Agent(
  role = "Senior Tech Researcher",
  goal = "Research about how the tech is build and other details about the {topic}",
  verbose = True,
  memory = True,
  backstory = ("""
      You are a veteran Tech Analyst who spent a decade analyzing viral metrics for top Silicon Valley publications. 
      You don't just read white papers; you live in the comment sections, Reddit threads, and Twitter replies to 
      decode exactly what triggers the audience. Your superpower is separating
      boring engineering jargon from the "spicy" details that drive clicks and debate.
    """),
allow_delegation = True,
)




# creating a tech writer agent
writing_agent = Agent(
  role = "Tech Writer",
  goal = "Write insightful and cool tech article on {topic}",
  verbose = True,
  memory = True,
  backstory = (
    """
    You are a Lead Tech Columnist who believes that "specs" are boring, but "experiences" are viral. 
    Your background bridges software engineering and creative non-fiction, giving you the unique ability 
    to turn dry technical data into compelling, 
    human-centric stories. You write with a "hook-first" mentality, 
    knowing that modern readers scan before they read.
    """
  ),
  allow_delegation = False,
)

research_task = Task(
  description=("""
    Execute a forensic, market-savvy research dig on {topic} that separates 
               the engineering reality from the marketing fluff.
    Be sure to identify the 'golden nuggets'—the under-appreciated 
               features or price-to-performance sweet spots—that the public appreciates most.
    Synthesize a raw, unvarnished profile of {topic} based on real-world
                user feedback and technical benchmarks.
 """),
 expected_output= ("A comprehensive 3-4 paragraphs long report on latest in tech."),
 agent = tech_researcher
)


writing_task = Task(
  description=("""
    Draft a punchy, culturally resonant narrative on {topic} that feels less like a manual and more like a conversation.           
    Compose a relatable, hook-driven article on {topic} that translates dry 
               specifications into a vivid user experience.
    Write a crystal-clear, high-value breakdown of {topic} that respects the reader's intelligence but kills the jargon
 """),
 expected_output= ("A concise and short title with brief one to two paragraph article on {topic}."),
 agent = writing_agent,
 async_execution=False,
 output_file="tech_article.md"
)

crew = Crew(
  agents=[tech_researcher, writing_agent],
  tasks = [research_task, writing_task],
  process=Process.sequential,
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=False
)

result = crew.kickoff(inputs={"topic" : "Claude Opus 4.5 in Software Engineering"})
print(result)