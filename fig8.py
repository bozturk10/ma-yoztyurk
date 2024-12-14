from scipy.spatial import distance
import pandas as pd
survey_population_df_multilabel=pd.read_csv('survey_population_df_multilabel.csv')
survey_population_df_multilabel.set_index('Unnamed: 0',inplace=True)
survey_population_df_multilabel.columns= [int(col) for col in survey_population_df_multilabel.columns]
waves_to_plot= [17, 18, 19, 20, 21]
wave_dates={12: '05-11-2019',
 13: '21-04-2020',
 14: '03-11-2020',
 15: '25-02-2021',
 16: '06-05-2021',
 17: '07-07-2021',
 18: '11-08-2021',
 19: '15-09-2021',
 20: '29-09-2021',
 21: '09-12-2021'}

def get_survey_to_survey_JS_distances(survey_population_df, fname='s2s_JS_dist.html', font_size=25):
    rs = []
    for wave_id in waves_to_plot:
        for col in survey_population_df:
            if col <= wave_id:
                js = distance.jensenshannon(survey_population_df[col], survey_population_df[wave_id])
                r = {
                    'd1': col,
                    'd2': wave_id,
                    'js': js
                }
                rs.append(r)
    df = pd.DataFrame(rs)

    df['text'] = "wave " + df['d1'].astype(str) #+ "<br>" + df['d1'].map(wave_dates)
    df['d2'] = df['d2'].astype(str)

    import plotly.graph_objects as go
    fig = go.Figure()

    # Add a trace for each category in d2
    for category in df['d2'].unique():
        category_df = df[df['d2'] == category]
        fig.add_trace(go.Scatter(
            x=category_df['d1'],
            y=category_df['js'],
            mode='lines+markers',
            name=str(category),
            text=category_df['text'],
            line=dict(width=8)
            
        ))

    # Update layout with adjustable font size
    fig.update_layout(
        width=1800,   # Increase width for landscape
        height=1200,   # Adjust height for better fit
        title=dict(text='', font=dict(size=font_size)),  # Empty title but font size is adjustable
        xaxis_title=dict(text='Wave', font=dict(size=font_size)),
        yaxis_title=dict(text='JS', font=dict(size=font_size)),
        xaxis_tickangle=45,
        xaxis_tickvals=df['d1'],
        xaxis_ticktext=df['text'].apply(lambda x: f"<b>{x}</b>"),
                yaxis=dict(
            tickmode='linear',
            ticktext=df['js'].apply(lambda x: f"<b>{x}</b>"),
            tick0=0,
            dtick=0.05,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=1,
            griddash='dash',  # Make the grid lines dashed
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=1,
        ),
                plot_bgcolor='white',
        font=dict(size=font_size),
        legend_font=dict(size=45),

    )

    # Save to HTML
    fig.write_html(fname)
    return fig

get_survey_to_survey_JS_distances(survey_population_df_multilabel,fname='ex2_s2s_JS_dist_multilabel.html',font_size=30)