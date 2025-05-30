import pandas as pd
import requests
import datetime
import json
from gensim.utils import simple_preprocess
from cleantext import clean


from langdetect import detect
from bs4 import BeautifulSoup

##########################################
##########################################


class scrape_steam_game:
    """
        def:
            the object aims to srcape the game reviews and guides given a steam game id
        
        Functions:
            Private:
                get_user_reviews()
                json_scrape()
                table_scrape()

                get_section_text()
                get_player_guides()

                fix_mojibake()
                translate_emojis_to_text()
                readable_description()

            
            Public:
                scrape_review_info()
                scrape_player_guides()

    """

    def __init__(self, game_id):
        self.game_id = game_id
        self.params = {
            "json": 1,
            "language" : 'english',
            "cursor": "*",
            "num_per_page": 100,   # retrive n reviews per request
            "filter": "recent"
        }

        
    
    ##########################################
    ##########################################

    # Scrape reviews

    # Private Function
    def __get_user_reviews(self, game_id, params):
        """
            def:
                scrape player reviews from a game
            
            parameters:
                game_id: a id of a game from steam
                params: setting of web scraping
            
            return:
                player_reviews in json format

        """
        user_review_url = f'https://store.steampowered.com/appreviews/{game_id}'
        req_user_review = requests.get(
            user_review_url,
            params = params
        )

        if req_user_review.status_code != 200:
            print(f'fail to get reponse. Status Code: {req_user_review.status_code}')
            return {"success": 2}
        
        try:
            players_reviews = req_user_review.json()
        except:
            return {"sucess": 2}
        
        return players_reviews

    
    def __json_scrape(self, game_id, params):
        player_reviews = []
   
        while True:
            reviews = self.__get_user_reviews(game_id = game_id, 
                                        params = params)

            # cannot be scraped
            if reviews["success"] != 1:
                print("Not a success")
                break
            
            if reviews["query_summary"]["num_reviews"] == 0:
                break
            
            review_list = reviews["reviews"]
            
            # iterate the review to check if they are in english
            for review in review_list:
                if self.__en_classifier(review["review"]):
                    # fix issues such as correct Canâ€™t to Can't and remove ' from string
                    review = review["review"].encode('cp1252', errors='replace').decode('utf-8', errors='replace').replace("’", "")
                    cleaned_review = self.__review_cleaning(review)
                    if cleaned_review != "":
                        player_reviews += [review]
            
            try:
                cursor = reviews["cursor"]
            except Exception as e:
                cusor = ''

            if not cursor:
                break
            
            params["cursor"] = cursor

        return player_reviews

    def __table_scrape(self, game_id, params):
        player_review_df = pd.DataFrame(columns=[
        "playtime_forever", "num_games_owned", "num_reviews",
        "votes_up", "votes_funny", "weighted_vote_score", 
        "comment_count", "steam_purchase",
        "written_during_early_access", "primarily_steam_deck",
        "timestamp_created", "review"
    ])

        while True:
            reviews = self.__get_user_reviews(game_id = game_id, 
                                        params = params)
            
            # cannot be scraped
            if reviews["success"] != 1:
                print("Not a success")
                break
            
            if reviews["query_summary"]["num_reviews"] == 0:
                break

            for review in reviews["reviews"]:

                # check if the review is english
                if self.__en_classifier(review["review"]):
                    text_review = review["review"].encode('cp1252', errors='replace').decode('utf-8', errors='replace').replace("’", "")
                    cleaned_review = self.__review_cleaning(text_review)
                    if cleaned_review == "":
                        continue
                
                    # Get the post time
                    time_stamp = review["timestamp_created"]
                    human_readable_date = datetime.datetime.fromtimestamp(time_stamp, tz=datetime.timezone.utc)
                    formatted_date = human_readable_date.strftime('%Y-%m-%d %H:%M:%S')

                    player_review_df.loc[len(player_review_df)] = {
                        "playtime_forever": review["author"]["playtime_forever"],
                        "num_games_owned": review['author']['num_games_owned'],
                        "num_reviews": review['author']['num_reviews'],
                        "votes_up": review['votes_up'],
                        "votes_funny": review['votes_funny'],
                        "weighted_vote_score": review['weighted_vote_score'],
                        "comment_count": review['comment_count'],
                        "steam_purchase": review['steam_purchase'],
                        "written_during_early_access": review['written_during_early_access'],
                        "primarily_steam_deck": review['primarily_steam_deck'],
                        "timestamp_created": formatted_date,
                        "review": cleaned_review
                    }
            
            try:
                cursor = reviews["cursor"]
            except Exception as e:
                cusor = ''

            if not cursor:
                print("Reached the end of all comments.")
                break
            
            params["cursor"] = cursor

        return player_review_df
    
    
    ##########################################
    ##########################################

    # Scrape Guides
    
    def __get_section_text(self, sections):
        """
            def: 
                get text from each section from html

            parameters:
                sections: sections in html

            return:
                guide: text from the page
        """
        guide = ""

        # Iterate each section to get text
        for section in sections:
                    title = section.find("div", class_="subSectionTitle")
                    content = section.find("div", class_="subSectionDesc")

                    title_text = title.get_text(strip=True) if title else "No Title"
                    content_text = content.get_text(separator="\n", strip=True) if content else "No Content"

                    if content_text.replace(" ", "") != "":
                        section_title = f"Section Title: {title_text}\n"
                        content = f"Content: \n{content_text}"
                        guide += section_title + content + "\n"
                    else:
                        break

        return guide

    def __get_player_guides(self):
        """
            def:
                get guides of the game for chatbot
            
            parameters:
                game_id: the id of the game from steam

            return:
                guides: a list of guides from the game on steam
        """
        link = f'https://steamcommunity.com/app/{self.game_id}/guides/?browsefilter=trend&filetype=11&requiredtags%5B0%5D=english&p=1'

        guides = []
        
        while True:
            # request access
            req_game_guide = requests.get(link)

            # Get html
            guide_soup = BeautifulSoup(req_game_guide.text, 'html.parser')

            current_page_num = int(link[-1])

            # Get links of guide
            hrefs = []
            for a_tag in guide_soup.find_all('a', href=True):
                if a_tag.find('div', class_='workshopItem'):
                    hrefs.append(a_tag['href'])
            
            # Get every guides on the page
            for href in hrefs:
                req_player_guide = requests.get(href)
                player_guide_soup = BeautifulSoup(req_player_guide.text, "html.parser")
                sections = player_guide_soup.find_all("div", class_ = "detailBox")

                # Get text
                guide = self.__get_section_text(sections)

                # append guide to a list
                if guide != "":
                    guides.append(guide)

            # find the cursor
            page_cursor = guide_soup.find_all('a', class_ = "pagebtn", href=True)
            next_page_num = int(page_cursor[-1]["href"][-1])


            # if the next page number is less than current page hum break the while loop 
            # otherwise, obtain the link of next page
            if next_page_num < current_page_num: break 
            else: link = page_cursor[-1]["href"]
        
        return guides
    
    ##########################################
    ##########################################

    # clean review

    def __review_cleaning(self, review):
        """
            def: clean latin letters or misread text
               return an empty string if less than 3 words after removing notations  
        """
        cleaned_review = simple_preprocess(clean(review))
        if len(cleaned_review) >= 3:
            return " ".join(cleaned_review)
        
        return ""
        
    ##########################################
    ##########################################

    # English Classifier

    def __en_classifier(self, review):
        """
            def:
                classify if a review is in english
            
            parameters:
                review: text data
            
            return:
                a boolean value
        """
        try:
            language = detect(review)
            return language == "en"
        except Exception:
            return False


    ##########################################
    ##########################################


    # Public function
    def scrape_review_info(self):
        """
            def: 
                a scrape reviews    given a game id on Steam
            
            return:
                returns a list of all player review infomation in tabl;e format from table_scrape() 
        """
        # return self.__json_scrape(game_id = self.game_id, 
        #                 params = self.params)

        return self.__table_scrape(game_id = self.game_id,
                            params = self.params)
    
    # Public Function
    def scrape_guides(self):
        """
            def: 
                a scrape guides given a game id on Steam
            
            return:
                returns a list of guide of a steam game
        """
        return self.__get_player_guides()
        