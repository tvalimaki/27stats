# -*- coding: utf-8 -*-
from scrapy.spiders import Spider
from scrapy.http import Request
import re


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
            route_info = {
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
            crag_link = row.css('.stxt a::attr(href)').extract()[1]
            crag_map = response.urljoin(crag_link) + '/cragmap'
            req = Request(crag_map, callback=self.parse_crag, dont_filter=True)
            req.meta['route_info'] = route_info
            yield req

    def parse_crag(self, response):
        route_info = response.meta.get('route_info')
        crag_location = response.css('.craglocation a::text').extract_first()
        if crag_location:
            crag_location = crag_location.split(',', 1)
            country = crag_location[-1].strip()
            area = crag_location[0].replace('in the area of ', '')
            route_info['country'] = country
            route_info['area'] = area
            m = re.search('"map":{"latitude":"([-.0-9]+)","longitude":"([-.0-9]+)"',
                          response.text)
            if m:
                route_info['lat'] = m.group(1)
                route_info['lon'] = m.group(2)
        else:
            route_info['country'] = ''
            route_info['area'] = ''
            route_info['lat'] = ''
            route_info['lon'] = ''
        yield route_info
