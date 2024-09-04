import openai
import re

# Replace with your actual OpenAI API key
openai.api_key = 'sk-proj-jgxQFYKoAYUmMTKdS9LCT3BlbkFJ0LZfcIgh1d2jNil9FOA7'

def check_resume_jd_match(resume, job_description):
    # Constructing the prompt
    prompt = f"""
    I have the following resume:

    {resume}

    And the following job description:

    {job_description}

    Please analyze the resume and the job description and determine how well they match. Provide a score out of 10 where 10 means an excellent match and 0 means no match at all. Additionally, provide a brief explanation for the score.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150  # You can adjust the number of tokens as needed
    )

    return response.choices[0].message['content'].strip()

def extract_score(response_text):
    # Extract the score from the response text
    match = re.search(r'\b(\d{1,2})\b', response_text)
    if match:
        return int(match.group(1))
    return None

def predict_resume_to_jd(resume, job_description):
    response_text = check_resume_jd_match(resume, job_description)
    score = extract_score(response_text)
    # Convert the score to a percentage out of 100
    score_percentage = score * 10 if score is not None else 0
    prediction = 1 if score_percentage >= 50 else 0
    return prediction, score_percentage, response_text

# Sample resume and job description
resume = """
resume mobile malladi prasad email objective obtain challenging php developer position enable utilize skill acquired knowledge responsibly contribute best knowledge effort growth company professional summary around year experience php moodle codeigniter wordpress mysql ajax leading web development software languagestools strong working knowledge php codeigniter wordpress mysql leading web development software languagestools basic working knowledge web development tool drupal bootstrap working knowledge html xml cs javascript ajax exposure leading open source technology flexible easily adjustable work environment excellent problem solving analytical skill excellent track record interpersonal skill professional approach team player good leadership quality academic profile graduationthe stream mechanical engineering aim college engineering affiliated jntukakinada mark year intermediate intermediate math physic chemistry sri gmc balayogi junior college affiliated board intermediate education ap mark year ssc ssc passed zph school affiliated board secondary education ap mark year professional skill programming language php data base management system mysql web technology html cs javascripts ajax jquery operating system windowslinux web server apache content management system wordpress framework codeigniter lm moodle others msoffice eclipse dreamweaver employment history working php developer commlab india llp auguest still date worked php developer empover itech pvt ltd november june worked php developer klesis global pvt ltd september september project summary lm project url environment php javascript ajax mysql bootstrap platform window overview project hindustan market india largest india trade directory provides bb service indian exporter directory seller exporter importer etc project url environment php codeigniter javascript ajax mysql platform window overview project huzzat app share visiting card social networking business connect project url environment html cs javascript ajax jquery java platform window overview project learn play app quiz questionsfill blank drag drop single multiple select question project url cognitivecarecom environment php html cs javascript ajax jquery mysql platform window overview project cognitivecare passionate bunch problem solver trying solve biggest complex problem healthcare project url environment php wordpress javascript ajax mysql platform linux overview project global christian news content management systemcms website maintenance various country news christian people project url environment php wordpress javascript ajax mysql platform linux overview project oxford center religion public life website used conference workshop topic enterprise solution poverty faith economic enterprise religion human right religious freedom advocacy family marriage project url environment php moodle javascript ajax mysql platform linux overview project oxford center religion public life website used online training class project url environment php wordpress javascript ajax mysql platform linux overview project isaac publishing ecommerce website using sale christian relative book various contries project url environment php codeigniter ajax javascript mysql bootstrap platform window overview project prkbloodservices blood donation website register site freely search selected blood group project url environment php drupal javascript ajax mysql platform linux overview project barnabasfund content management fund donate website christianity particular project project url environment codeigniter javascript ajax mysql platform window overview project brenterior completely content management systemcms backend admin mange content u service portfolio testimonial comment client newsroom contact usadmin manage page content admin add slider backend admin manage gallery widget setting project url project url project url project url personal profile date birth nationality indian gender male marital status single language known english telugu declaration hereby declare particular mentioned true best knowledge place hyderabad prasad date"""

job_description = """
title java developer looking java developer experience building highperforming scalable enterprisegrade application part talented software team work mission critical application java developer role responsibility include managing javajava ee application development providing expertise full software development lifecycle concept design testing java developer responsibility include designing developing delivering highvolume lowlatency application missioncritical system responsibility contribute phase development lifecycle write well designed testable efficient code ensure design compliance specification prepare produce release software component support continuous improvement investigating alternative technology presenting architectural review bsms degree computer science engineering related subject proven handson software development experience proven working experience java development hand experience designing developing application using java ee platform object oriented analysis design using common design pattern profound insight java jee internals classloading memory management transaction management etc excellent knowledge relational database sql orm technology jpa hibernate experience spring framework experience sun certified java developer experience developing web application using least one popular web framework jsf wicket gwt spring mvc experience testdriven development
"""

# Check the match
prediction, score_percentage, response_text = predict_resume_to_jd(resume, job_description)
print(f"Prediction: {prediction}")
print(f"Matching Score: {score_percentage}%")
print(f"Explanation: {response_text}")
