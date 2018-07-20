# 27 Stats

A set of tools to scrape and visualize a user's tick list from [27 Crags](https://www.27crags.com).

![A climbing odyssey](a_climbing_odyssey.png)

## Installation

Install to a virtual environment using virtualenvwrapper.

```bash
mkproject 27stats
git clone https://github.com/tvalimaki/27stats.git .
pip install -r requirements.txt
```

## Usage

To scrape a user's tick list, display, and save the final visualization as a `.png` file, run

```bash
python py27stats.py -s username
```

The tick list must be public. See `python py27stats.py -h` for other options.

To just scrape a user's tick list and save it as a `.csv` file, the scraper can be run independently

```bash
scrapy crawl 27crags -a user=username
```

## License

Released under the [MIT License](LICENSE).