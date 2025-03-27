from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def google_search_selenium(query):
    # 웹드라이버 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 브라우저 화면 표시 안함
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # 구글 검색 페이지 접속
        driver.get('https://www.google.com')
        
        # 검색어 입력
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        # 잠시 대기
        time.sleep(2)
        
        # 검색 결과 추출
        results = driver.find_elements(By.CSS_SELECTOR, 'div.g')
        
        search_results = []
        for result in results[:5]:  # 상위 5개 결과
            try:
                title = result.find_element(By.CSS_SELECTOR, 'h3').text
                link = result.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
                snippet = result.find_element(By.CSS_SELECTOR, 'div.VwiC3b').text
                
                search_results.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet
                })
            except Exception as e:
                print(f"결과 추출 중 오류: {e}")
        
        return search_results
    
    except Exception as e:
        print(f"검색 중 오류: {e}")
        return []
    
    finally:
        driver.quit()

def main():
    fitness_levels = [
        "초보자 운동 루틴",
        "중급자 운동 루틴",
        "상급자 운동 루틴"
    ]
    
    for level in fitness_levels:
        print(f"\n===== {level} 검색 결과 =====")
        
        # Selenium으로 검색 결과 크롤링
        search_results = google_search_selenium(level)
        
        # 결과 출력
        for i, result in enumerate(search_results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"링크: {result['link']}")
            print(f"요약: {result['snippet']}")

if __name__ == "__main__":
    main()