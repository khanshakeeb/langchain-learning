from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Michael Joseph Jackson (* 29. August 1958 in Gary, Indiana; † 25. Juni 2009 in Los Angeles, Kalifornien) war ein US-amerikanischer Pop-, Soul-, R&B-, Funk-, Disco- und Rocksänger, Tänzer und Songwriter. Er arbeitete auch als Musik- und Filmproduzent sowie Musikmanager. Er ist mit über 500 Millionen verkauften Tonträgern einer der kommerziell erfolgreichsten Musiker und erhielt den Beinamen „King of Pop“. Mit Tänzen wie dem Moonwalk und seinen Musikvideos prägte er auch die visuelle Popkultur nachhaltig.

Als achtes Kind der Jackson-Familie trat er bereits 1966 als Siebenjähriger mit den Jackson 5 auf. Der internationale Durchbruch gelang ihm 1982 mit dem Soloalbum Thriller, das zum bisher meistverkauften Album wurde. Zu seinen weiteren Erfolgen zählen Bad, Dangerous und HIStory, die alle zu den Top 100 der meistverkauften Alben gehören.

Ab den späten 1980ern geriet Jackson zunehmend wegen seines veränderten Aussehens, seines exzentrischen Verhaltens und wegen Missbrauchsvorwürfen in die Schlagzeilen. Er starb 2009 an einer Sedativa-Überdosis; der verantwortliche Arzt Conrad Murray wurde wegen fahrlässiger Tötung zu einer Haftstrafe verurteilt. Jacksons Tod löste weltweite Anteilnahme aus – über eine Milliarde Menschen verfolgten die Trauerfeier in den Medien.
    """
    summary_template = f"""
    given the information, {information}, about a person I want to create:
    1. A short summary
    2. Two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
