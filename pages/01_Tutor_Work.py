import streamlit as st

def main():
    st.title('Tutor Work')

    st.markdown('''
        
        #### Introduction
                
        [Link to Tutor Website](https://tasks-app.streamlit.app)
                
        ***Login**: demo*
                
        ***Password**: demo_user_2024Q*
                
        You are welcome to this section! As I mentioned before, my work for quite some time has been to be a great tutor. In this section, I want to describe how my work has evolved in recent years. Since the time I really got into programming, I've tried to implement new things in the educational process to achieve two main goals:

        - To practice the new things that I discovered 🤓;
        - To automate the process of issuing homework and class assignments, as WhatsApp or Telegram screenshots are not so effective;
        - To make the educational process more attractive, interesting, and productive for my students 🤓.

        #### Technical details

        Taking these goals into account, I decided to create a web application using *Streamlit*. The idea was straightforward: add several tasks in each category and instruct students on which ones to complete.

        To achieve these goals, the following steps were made:

        - Find a good database of tasks on the internet;
        - Write a scraper to get them in a convenient format or to convert them into one;
        - Implement them into a *Streamlit* application.

        Initially, I scrapped a lot of tasks from different sources into a *.txt* format. Whenever someone wanted to get a task, it was scanned through this file and returned. However, this method didn't look nice, and the solution was to put all these tasks into a more appropriate container—a database. Since I was learning both *MySQL* and *MongoDB* at that time, I decided to try both (of course, *MySQL* is quicker to read information from, but for practice, implementing both was quite fun).

        The next question was how to connect these local databases with a web application on Streamlit, and I decided to use *Amazon Web Services*:

        - For *MongoDB*, an *EC2* instance was run, and the server with *MongoDB* was initialized. Data from the local *MongoDB* server was transferred to my *EC2*.
        - For *MySQL*, I used *AWS RDS* since it allows easy operation over data. Data was also transferred from the local *MySQL* server to my *RDS*.
                
        There was a problem with images that are presented in some tasks. To make it more accessible with the help of scrapper I automatically download these images for correspondig tasks and
        right after upload to my own *AWS S3 bucket* with the right name. So when one open the task, it knows which image from which place in my S3 to take.

        To make the work more engaging, I added a section for Statistics, or rather, simple visualizations with *matplotlib*, where students can see how many tasks are already done and how many are left. It is always fun to see not only the texts of tasks but also some plots.

        For the Informatics section, a special field for answer checking was also added, so students wouldn't have to wait until the next lesson to know the right answer and could get a 'Yes' or 'No' response immediately.

        **Homeworks** is of *.pdf* format from *LaTex* document. I wrote a script in Python and bash to make it possbile automatically upload the newest version
        of it directly to the website(local machine -> google drive with publich access and github account -> pdf container in streamlit app) and give an easy access to it.
        
        #### Access
                        
        Access to the website is **only possible** with a login and password 🔒, which can only be obtained personally from me.

        If it interests you, you can use the link, login, and password at the beginning of this section to take a look at this app.

        #### Conclusion

        After all the changes that have been made, I see that students are more inclined to study now, and all my efforts were definitely not a waste of time.

        For me, in general, it is always fun to come up with an idea and then try to implement it, especially if it can be useful for others.

    ''')

if __name__ == '__main__':
    main()