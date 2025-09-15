#데이터 수집 - 라이브러리 설치 pip install bing_image_downloader

# celeb = ["존박", "peter crouch", "Donald trump"]

from bing_image_downloader import downloader


downloader.download(
    "피카츄",
    limit=100,
    output_dir = r"week_4\data",
    adult_filter_off=True,
    force_replace=True,
    timeout=60
)