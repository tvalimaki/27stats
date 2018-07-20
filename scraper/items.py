# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class Route(scrapy.Item):
    date = scrapy.Field()
    name = scrapy.Field()
    stars = scrapy.Field()
    comments = scrapy.Field()
    crag = scrapy.Field()
    type = scrapy.Field()
    grade = scrapy.Field()
    ascent = scrapy.Field()
