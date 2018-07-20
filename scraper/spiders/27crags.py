# -*- coding: utf-8 -*-
from scrapy.spiders import Spider
from scrapy.http import Request


class CragSpider(Spider):
    name = '27crags'
    allowed_domains = ['27crags.com']

    def start_requests(self):
        yield Request('https://27crags.com/climbers/%s/ascents/all' % self.user)

    def parse(self, response):
        for row in response.css('.col-md-12 tbody tr'):
            route, crag = row.css('.stxt a::text').extract()[:2]
            ascent = row.css('.ascent-type::text').extract_first().strip().split('\n')
            route_type = row.css('td')[4].css('::text').extract_first().strip()
            if route_type == 'Partially...':
                route_type = 'Partially bolted'
            if len(ascent) > 1:
                ascent_type = ascent[0]
                ascent_style = ascent[1]
            else:
                ascent_type = ''
                ascent_style = ascent[0]
            yield {
                'date': row.css('.ascent-date::text').extract_first().strip(),
                'route': route,
                'stars': len(row.css('div.star').extract()),
                'comments': row.css('.ascent-details::text').extract_first().strip(),
                'crag': crag,
                'type': route_type,
                'grade': row.css('span.grade::text').extract()[-1].strip(),
                'ascent_style': ascent_style,
                'ascent_type': ascent_type,
            }
