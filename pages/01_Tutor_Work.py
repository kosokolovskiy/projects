import streamlit as st

def main():
    st.title('Tutor Work')

    st.markdown('''
        
        #### Introduction
                
        [Link to Tutor Website](https://tasks-app.streamlit.app)
                
        *Login*: demo
                
        *Password*: demo_user_2024Q
                
        You are welcome to this section! As I mentioned before, my work since quite a long time - to be a great Tutor.
        In this section I want to describe a little, how my work has evolved in recent years. Since the time I really got
        into Programming, I tried to implement new things in educational process to achieve two goals:
                
        - to practice those new things that I discovered 🤓;
                
        - to automate 🤖 the process of issuinng homework and class assignments (since WhatsApp or Telegram screenshots are not so good);

        - to make educational process more attractive, interesting and productive for my students 🤓.

        #### Technical details
                
        Taking these three goals into account, I decided to try for the first time to make an Web Application with *Streamlit*.
        The idea was pretty easy: add several tasks in each category and tell students which one of them they should do.
        
        To achieve this goals the following steps were made:
            
        - find in the internet good database of tasks;

        - writer scrapper to get them in the convenient format or to convert them into;
                
        - implement them into *Streamlit* application.

        Firstly, I scrapped a lot of tasks from different sources into *.txt* format and everytime one wanted to get the task, it was scanned through this file
        and returned back. However, it didn't look nice and the solution was to put all these tasks into the more appropriate container - 
        database. Since I was learning at that time both: *MySQL* and *MongoDB*, I decided to try both(of course, *MySQL* is quicker to 
        read information, but for practice it was pretty fun to implement both).

        The second question was: how to connect these local databases with web application on streamlit and I decided to go to 
        *Amazon Web Services*:
                
        - for *MongoDB* *EC2* instance was run and the server with *MongoDB* initialized. Data from local *MongoDB* server was transferred to my *EC2*.

        - for *MySQL* I use *AWS RDS*, since it allows to easily operate over data. Data was also from local *MySQL* server transferred to my *RDS*.
                
        To make the work more colorful, I added the section of Statistics, or rather simple visualization with *matplotlib*, where the student can see how 
        many tasks are already done and how many are left. It is always fun to see not only texts of tasks, but also some plots. 
                
        For the section of *Informatics*, the special field for answer checking also was added, not to wait till the next lesson to know the 
        right answer and get the result 'Yes' or 'No' immediately. 
                
        Also access to the website is **only possible** with login and password 🔒, that is only possible to get personally from me. 
                
        #### Conclusion
                
        After all the changes that has been done, I see that students are more inclined to study now and all my efforts definitely were not waste of time.
                
        For me, in general,  it is always fun to come up with an idea and then try to implement it, especially if it can be useful for the others.

    ''')

if __name__ == '__main__':
    main()