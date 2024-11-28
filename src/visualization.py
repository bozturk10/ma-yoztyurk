import plotly.graph_objects as go


def plot_comparison_chart(data, title, output_file_path,save=False):
    # Create traces for Actual and Synthetic data
    trace1 = go.Bar(
        x=data['kpx_840_class1_name'], 
        y=data['survey_proportion'], 
        name='survey_proportion',
        marker=dict(color='RoyalBlue')
    )
    trace2 = go.Bar(
        x=data['kpx_840_class1_name'], 
        y=data['llm_proportion'], 
        name='llm_proportion',
        marker=dict(color='Crimson')
    )

    fig = go.Figure(data=[trace1, trace2])

    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,  # Rotate labels to prevent overlap
        xaxis_title='Category',
        yaxis_title='Percentage (log scale)',
        barmode='group',  # Group bars for side-by-side comparison
        legend_title_text='Type'
    )
    if save and output_file_path.endswith(".html"):    
        # Save the figure as an HTML file
        fig.write_html(output_file_path)
    elif save and output_file_path.endswith(".png"):
        fig.write_image(output_file_path + ".png")
    fig.show()

