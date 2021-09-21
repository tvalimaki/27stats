import os
import argparse
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')
sns.set_palette(sns.xkcd_palette(
    ['greyish', 'pale red', 'amber', 'dark grey']))

# Grade categories [3, 3+, ..., 9c+]
gradeType = CategoricalDtype(
    [grade + plus for grade in
     [str(number) for number in range(3, 6)] +
     [str(number) + sub for number in range(6, 10) for sub in 'abc']
     for plus in ['', '+']], ordered=True
)

# Ascent style categories
styleType = CategoricalDtype(
    ['Top rope', 'Red point', 'Flash', 'On-sight'], ordered=True
)

# Style for figure titles
titleSpec = {'fontsize': 14, 'color': 'C0', 'weight': 'bold'}


def crawl(username):
    """Crawl 27 Crags for the user's tick list

    Args:
        username (str): Name of user
    """
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    process = CrawlerProcess(get_project_settings())
    process.crawl('27crags', user=username)
    process.start()


def create_plot(data, xticks, ylim, **kwargs):
    """Create a scatter and yearly top10 avg. plot for the ticks in data,
    along with a marginal histogram.

    Args:
        data (DataFrame): A pandas DataFrame containing the ticks
        xticks (DatetimeIndex): Locations of xticks
        ylim (tuple): Limits for the y-axis (grade cat codes)
        **kwargs: Not used but needed for calls from FacetGrid
    """
    ax = plt.gca()
    # plot datapoints
    sns.stripplot(x='date_ordinal', y='grade', data=data,
                  alpha=.3, hue='ascent_style', jitter=True, ax=ax)

    # plot marginal histogram
    divider = make_axes_locatable(ax)
    marg = divider.append_axes('right', '15%', pad='3%', sharey=ax)
    a = data.groupby(['ascent_style', 'grade'])['route'].count().reset_index()
    left = np.zeros(len(gradeType.categories),)

    for name, df in a.groupby('ascent_style'):
        marg.barh(df.grade.cat.codes, df.route, left=left[df.grade.cat.codes],
                  linewidth=0)
        left[df.grade.cat.codes] += df.route

    for grade in np.unique(data.grade.cat.codes):
        marg.text(left[grade], grade, ' %d' % left[grade], color='C0',
                  verticalalignment='center', horizontalalignment='left',
                  fontsize=8)

    marg.axis('off')

    # get yearly top10 average
    top10 = data.groupby(['ascent_style', 'year'], as_index=False).apply(
        lambda x: x.nlargest(min(10, len(x)), 'grade_ordinal'))
    top10 = top10.groupby(['ascent_style', 'year']).agg(
            {'grade_ordinal': np.mean, 'date_ordinal': np.mean}).rename(
                columns={'grade_ordinal': 'top10'})

    # plot interpolation of yearly top10
    for name, df in top10.groupby('ascent_style'):
        df.dropna(inplace=True)
        if len(df) < 2:
            continue
        if len(df) == 2:
            kind = 'linear'
        elif len(df) == 3:
            kind = 'quadratic'
        else:
            kind = 'cubic'
        new_index = np.arange(df.date_ordinal.iloc[0],
                              df.date_ordinal.iloc[-1])
        f = interp1d(df.date_ordinal.values, df.top10, kind=kind)
        df = pd.DataFrame()
        df['grade'] = f(new_index)
        df.index = new_index
        df.plot.line(
            ax=ax,
            color='C%s' % pd.Series(name).astype(styleType).cat.codes[0]
        )

    # set axis properties
    sns.despine(ax=ax, left=True, bottom=True)

    xtick_loc = [d.toordinal() for d in xticks]
    ax.set_xlim((xtick_loc[0], xtick_loc[-1]))
    ax.set_xticks(xtick_loc)
    ticklabels = list(xticks.year.values)
    ticklabels[-1] = ''
    ax.set_xticklabels(ticklabels)
    ax.xaxis.grid(False)

    ax.invert_yaxis()
    ax.set_ylim(ylim)
    ax.yaxis.grid(True)


def visualize(ticks):
    """Create visualizations from the tick list in file

    Args:
        ticks (DataFrame): The preprocessed tick list

    Returns:
        tuple: The produced matplotlib figures
    """
    xticks = pd.date_range(start='1/1/%s' % ticks.date.iloc[0].year,
                           end='1/1/%s' % str(ticks.date.iloc[-1].year+1),
                           freq='YS')
    ylim = (min(ticks.grade.cat.codes)-0.5,
            max(ticks.grade.cat.codes)+0.5)

    # create a title figure
    title, ta = plt.subplots(1, 1, figsize=(10, 1))
    ta.text(0.5, 0.5, 'A CLIMBING ODYSSEY', titleSpec, fontsize=24, color='C2',
            horizontalalignment='center', verticalalignment='center',
            transform=ta.transAxes)
    ta.axhline(0, color=titleSpec['color'])
    ta.axis('off')
    plt.tight_layout()

    # plot all ascents
    g = sns.FacetGrid(ticks, dropna=False)
    g.map_dataframe(create_plot, ylim=ylim, xticks=xticks)
    t = g.fig.suptitle('ALL ASCENTS', fontsize=titleSpec['fontsize'],
                       weight=titleSpec['weight'])
    t.set_color(titleSpec['color'])
    g.fig.set_size_inches(10, 4)

    # add legend
    ax = g.axes.flat[0]
    handles, labels = ax.get_legend_handles_labels()
    line = matplotlib.lines.Line2D([], [], color=sns.xkcd_rgb['greyish'])
    ax.legend(handles[:-5:-1] + [line], labels[:-5:-1] + ['Top 10 avg.'],
              bbox_to_anchor=(-0.07, 1), loc=1, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.87, right=0.97)

    # plot by ascent type
    ticks_type = ticks[ticks.type.isin(['Boulder', 'Sport', 'Traditional'])]
    g2 = sns.FacetGrid(ticks_type, col='type', dropna=False)
    g2.map_dataframe(create_plot, ylim=ylim, xticks=xticks)
    g2.set_titles('{col_name}')
    g2.set_xticklabels(rotation=45)
    t = g2.fig.suptitle('ASCENTS BY GENRE', fontsize=titleSpec['fontsize'],
                        weight=titleSpec['weight'])
    t.set_color(titleSpec['color'])
    g2.fig.set_size_inches(10, 3.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.77, right=0.97)

    # plot summary
    stats = {
        'Ascents total': ticks.shape[0],
        'Crags visited': len(ticks.crag.unique()),
        'Countries visited': len(ticks.country.unique()),
        'Days out': len(ticks.date.unique()),
        'FA': sum(ticks.ascent_type == 'FA'),
        '2nd': sum(ticks.ascent_type == '2nd'),
    }

    fig, ax = plt.subplots(1, 3, figsize=(10, 1.7))
    ax[0].text(0, 1, 'HIGHLIGHTS', titleSpec,
               verticalalignment='top', transform=ax[0].transAxes)
    ax[0].text(0, 0.8, "\n".join(stats.keys()), verticalalignment='top',
               transform=ax[0].transAxes)
    ax[0].text(0.7, 0.8, "\n".join(['%s' % v for v in stats.values()]),
               verticalalignment='top', transform=ax[0].transAxes)
    ax[0].axis('off')

    types = ticks.groupby('type')['route'].count()
    _, _, a = ax[1].pie(types.values, labels=types.index, autopct='%1.f%%',
                        startangle=90, counterclock=False,
                        wedgeprops={'linewidth': 0})
    for t in a:
        t.set_color('white')
    ax[1].axis('equal')

    style = ticks.groupby('ascent_style')['route'].count()
    style.index = style.index.remove_unused_categories()
    _, _, a = ax[2].pie(style.values[style.values > 0],
                        labels=style.index[style.values > 0],
                        colors=['C{}'.format(c)
                                for c in style.index[style.values > 0].codes],
                        autopct='%1.f%%',
                        startangle=90, counterclock=False,
                        wedgeprops={'linewidth': 0})
    for t in a:
        t.set_color('white')
    ax[2].axis('equal')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.03)

    return title, g.fig, g2.fig, fig


def visualize_map(ticks):
    """Visualize the crag locations on a world map

    Args:
        ticks (DataFrame): Preprocessed ticklist data

    Returns:
        Figure: The produced matplotlib figure
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # plot the world map
    map_crs = ccrs.AlbersEqualArea()
    fig, ax = plt.subplots(1, 1, figsize=(10, 3),
                           subplot_kw={'projection': map_crs})
    ax.add_feature(cfeature.LAND, facecolor='C0')
    ax.add_feature(cfeature.BORDERS, edgecolor='white', alpha=0.4)

    ax.spines['geo'].set_visible(False)
    ax.gridlines()

    # plot ascents as a scatterplot where more ascents at the same crag
    # produces a bigger blob
    db = ticks.pivot_table(index=['lat', 'lon'], values='route',
                           aggfunc='count').reset_index()

    sns.scatterplot(data=db, x='lon', y='lat', alpha=0.5, color='C1',
                    size='route', sizes=(20, 200), ax=ax, legend=False,
                    transform=ccrs.PlateCarree())

    # set the viewlimits of the map to match the figure aspect ratio
    top = 0.9
    left = 0.03
    right = 0.97
    bottom = 0.15
    size = fig.get_size_inches()
    width = (right - left)*size[0]
    height = (top - bottom)*size[1]

    old_width = ax.viewLim.width
    new_width = ax.viewLim.height*width/height
    width_change = new_width - old_width
    ax.viewLim.x0 = ax.viewLim.x0 - width_change/2.
    ax.viewLim.x1 = ax.viewLim.x1 + width_change/2.

    plt.subplots_adjust(top=top, left=left, right=right, bottom=bottom)

    return fig


def preprocess_data(file):
    """Preprocess the data from the tick list in file

    Args:
        file (str): Path to the file containing the tick list

    Returns:
        DataFrame: The processed tick list
    """
    ticks = pd.read_csv(file, parse_dates=['date'],
                        converters={'grade': str.lower,
                        'ascent_type': str})
    ticks = ticks.sort_values(by='date')
    ticks = ticks.replace('Partially bolted', 'Traditional')
    ticks.grade = ticks.grade.astype(gradeType)
    ticks.ascent_style = ticks.ascent_style.astype(styleType)
    ticks['date_ordinal'] = [date.toordinal() for date in ticks.date]
    ticks['year'] = [date.year for date in ticks.date]
    ticks['grade_ordinal'] = ticks.grade.cat.codes

    return ticks


def main():
    parser = argparse.ArgumentParser(
        description="Scrape and visualize a user's tick list from 27 Crags."
    )
    parser.add_argument('username', type=str,
                        help='name of the user whose tick list to visualize')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force crawl even if tick list exists')
    parser.add_argument('-s', '--save', action='store_true',
                        help='save figures to disk')
    parser.add_argument('-m', '--map', action='store_true',
                        help='plot ascents on a world map')
    args = parser.parse_args()

    file = './%s.csv' % args.username
    if args.force or not os.path.isfile(file):
        if os.path.isfile(file):
            os.remove(file)
        crawl(args.username)

    data = preprocess_data(file)
    fig = visualize(data)

    if args.map:
        fig += (visualize_map(data),)

    if args.save:
        from PIL import Image

        for i, f in enumerate(fig):
            f.savefig('fig{}.png'.format(i), dpi=300)
        img = []
        for i in range(i+1):
            img.append(Image.open('fig{}.png'.format(i)))
        width = img[0].size[0]
        height = [i.size[1] for i in img]

        graphic = Image.new('RGB', (width, sum(height)))
        h = 0
        for i in img:
            graphic.paste(i, (0, h))
            h += i.size[1]
        graphic.show()
        graphic.save('a_climbing_odyssey.png')
    else:
        plt.show()


if __name__ == '__main__':
    main()
