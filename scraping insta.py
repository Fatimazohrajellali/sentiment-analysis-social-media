# Imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import csv

# Configuration de Selenium
driver_path = "E:\\driver\\chromedriver.exe"
service = webdriver.chrome.service.Service(driver_path)
driver = webdriver.Chrome(service=service)

# Ouvrir Instagram
driver.get("http://www.instagram.com")

# Se connecter à Instagram
username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='username']")))
password = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='password']")))

username.clear()
username.send_keys("")  # Remplacez par votre nom d'utilisateur
password.clear()
password.send_keys("")  # Remplacez par votre mot de passe

# Cliquer sur le bouton de connexion
button = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()

# Cliquer sur "Plus tard" pour ignorer l'enregistrement des informations de connexion
not_now = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//div[contains(text(), "Plus tard")]'))).click()

# Rechercher un hashtag
searchbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//input[@aria-label='Saisie de la recherche']")))
searchbox.clear()

# Définir le hashtag à rechercher
keyword = "#Fear"
searchbox.send_keys(keyword)

# Attendre les résultats de la recherche
time.sleep(2)

# Cliquer sur le résultat du hashtag
hashtag_link = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, f"//a[contains(@href, '/{keyword[1:]}')]")))
hashtag_link.click()

# Attendre que la page se charge
time.sleep(5)

# Ouvrir un fichier CSV pour stocker les commentaires
with open('instagram_comments.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Commenter", "Comment"])  # En-têtes du CSV

    # Scraper les posts
    posts = driver.find_elements(By.XPATH, "//img[contains(@class, 'x5yr21d xu96u03 x10l6tqk x13vifvy x87ps6o xh8yej3')]")  # XPath des posts

    for post in posts:
        try:
            # Attendre que le post soit cliquable
            post = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//img[contains(@class, 'x5yr21d xu96u03 x10l6tqk x13vifvy x87ps6o xh8yej3')]")))

            # Faire défiler la page jusqu'à l'élément
            driver.execute_script("arguments[0].scrollIntoView();", post)

            # Cliquer sur le post en utilisant ActionChains pour éviter les problèmes de superposition
            actions = ActionChains(driver)
            actions.move_to_element(post).click().perform()

            time.sleep(2)

            # Récupérer l'URL du post
            post_url = driver.current_url

            # Récupérer les commentaires
            comments = []
            commenters = []

            comment_elements = driver.find_elements(By.XPATH, "//div[@class='x78zum5 xdt5ytf x1iyjqo2 xs83m0k x2lwn1j x1odjw0f x1n2onr6 x9ek82g x6ikm8r xdj266r x11i5rnm x4ii5y1 x1mh8g0r xexx8yu x1pi30zi x18d9i69 x1swvt13']//ul//li")
            for comment in comment_elements:
                try:
                    commenter = comment.find_element(By.XPATH, ".//a[contains(@class, 'x1i10hfl xjqpnuy xa49m3k xqeqjp1 x2hbi6w xdl72j9 x2lah0s xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r x2lwn1j xeuugli x1hl2dhg xggy1nq x1ja2u2z x1t137rt x1q0g3np x1lku1pv x1a2a7pz x6s0dn4 xjyslct x1ejq31n xd10rxx x1sy0etr x17r0tee x9f619 x1ypdohk x1f6kntn xwhw2v2 xl56j7k x17ydfre x2b8uid xlyipyv x87ps6o x14atkfc xcdnw81 x1i0vuye xjbqb8w xm3z3ea x1x8b98j x131883w x16mih1h x972fbf xcfux6l x1qhh985 xm0m39n xt0psk2 xt7dq6l xexx8yu x4uap5 x18d9i69 xkhd6sd x1n2onr6 x1n5bzlp xqnirrm xj34u2y x568u83')]").text
                    comment_text = comment.find_element(By.XPATH, ".//span[contains(@class, '_ap3a') and contains(@class, '_aaco')]").text
                    comments.append(comment_text)
                    commenters.append(commenter)

                    # Écrire les commentaires dans le fichier CSV
                    writer.writerow([commenter, comment_text])
                except Exception as e:
                    print(f"Erreur lors de la récupération d'un commentaire : {e}")
                    continue

            # Fermer le post
            close_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.x1i10hfl.x972fbf[role='button'] svg[aria-label='Fermer']")))
            close_button.click()
            time.sleep(1)

        except Exception as e:
            print(f"Erreur lors du scraping : {e}")
            continue

# Fermer le navigateur
driver.quit()
print("Scraping terminé. Les commentaires ont été enregistrés dans 'instagram_comments.csv'.")