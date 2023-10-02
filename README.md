# WilliamThesisMLCodeUCG
**
In a rapidly evolving information landscape, detecting and regulating misleading online
content (such as clickbait) is crucial for combating disinformation and for maintaining
trust in society. This study hypothesizes that an effective clickbait detection tool can be
developed through AI-human collaboration, providing high accuracy and rapid
execution. To test this hypothesis, a program was created using RandomForest Classifier,
a supervised learning algorithm, to tag a whole webpage of headlines as 'non-clickbait' or
'clickbait', with just a URL input. The tool achieved a test accuracy exceeding 94% and
demonstrated its utility by analyzing and storing data from various web pages in
real-time. Additionally, generative Natural Language Processing (NLP) techniques were
used to predict clickbait based on headlines and their paired articles. The analysis
revealed significant differences in the clickbait proportion across sampled British news
websites, highlighting an ongoing trend towards tabloidisation. These findings
underscore the potential of AI for automating subjective analysis and providing
real-time 'fact-checking', but also stress the necessity of human oversight for refining
algorithms. Whilst this study contributes to understanding clickbait usage in online
news in a number of aspects, its main finding is the potential of AI-Human
collaboration to fine-tune AI for real-time assessment. Although beyond the study, the
development of real-time monitoring metrics and tools would be a simple and obvious
next stop, providing feedback on ‘quality’ to users, publishers and regulators.


1. Introduction

1.1 Context
This study examines features of clickbait news headlines found on prominent UK news
websites, with an aim of discerning whether the traditional distinction between 'quality
press' and 'tabloid press' (the latter being synonymous with sensationalised news) is
echoed in these web headlines. For clarity, 'quality press' refers to news outlets
traditionally seen as providing more in-depth and balanced coverage. In contrast,
'tabloid press' denotes those historically associated with more sensationalist and
sometimes less rigorous reporting. In this context, clickbait refers to headlines that
exploit human curiosity and emotional responses to drive online traffic, often at the
expense of accuracy or context.
This study employs machine learning, sentiment analysis, and Natural Language
Processing (NLP) techniques to analyse the headlines from both 'quality press' and
‘tabloid press’ outlets. It hypothesises that both 'quality' and 'tabloid' presses employ
clickbait tactics to varying degrees. This hypothesis is informed by recent shifts in digital
journalism and the growing competition for online attention.
The study's methodology involves verifying news headlines and their accompanying
stories, subsequently categorising them as "non-clickbait" (essentially news/truth) and
"clickbait" (commercially-driven sensationalised-truth). Furthermore, it analyses the
positioning of these stories on the webpage and the fluctuation of the clickbait ratio
((Number of clickbait headlines)/(Number of non-clickbait headlines)) throughout a
24-hour cycle across different internet publications.
The research also encompasses the analysis of potential clickbait indicators, such as
emotional sentiment, to uncover the strengths and weaknesses of AI in this context. The
ultimate goal is not to replace human input but rather to complement it. AI can process
large quantities of data rapidly and objectively, while humans can fine-tune the model to
increase its accuracy. This collaborative approach can mitigate the slow pace and
potentially inconsistent subjectivity of human-only analysis. Human-only approaches
would simply be unable to sift and assess the volume of material in a timely manner,
leading to poor reporting standards being unaddressed.
In addition to its practical implications, this research holds significance for both
Computer Science and Social Science disciplines. It offers insights into the effectiveness
of machine learning models in analysing real-time clickbait data, as well as findings from
clickbait analysis concerning news web pages. The integration of these insights aims to
provide a comprehensive understanding of clickbait traits across different news websites.
The findings will inform future research, journalistic practices, and media literacy
initiatives. 
