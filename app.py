#imports :- 
import pandas as pd
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
from datetime import date
import os
import io
import math 
from flask import Flask, request , send_from_directory , jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

#starting the Flask
app = Flask(__name__)
CORS(app)
app.config["STATIC_FOLDER"]="./static"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/upload-csv', methods=['POST'])
async def upload_csv():
    # Define the path for the generated PDF
    pdf_path = './static/example.pdf'

    # Check if the 'plots' directory already exists
    dir_path = os.path.exists('./static/plots')

    # If a PDF file already exists, remove it
    if os.path.exists(pdf_path):
        os.remove("./static/example.pdf")

    # If the 'plots' directory does not exist, create it
    if not dir_path:
        os.mkdir('./static/plots')

    # Check if the 'file' part is in the request
    if 'file' not in request.files:
        print('file not found')
        return jsonify({'message': 'No file part in the request'}), 400

    # Get the file from the request
    file = request.files['file']

    # If the uploaded file has no filename, return an error response
    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    # Secure the filename to prevent security risks
    filename = secure_filename(file.filename)
    print("done save")

    # Save the uploaded file to the 'static' directory with its original name
    file.save(os.path.join('./static/' + filename))

    # Read the data from the CSV file into a Pandas DataFrame
    data = pd.read_csv(os.path.join('./static/' + filename))
    df = pd.DataFrame(data)

    # Get the event loop for asynchronous processing
    loop = asyncio.get_event_loop()

    # Call the File_maker function asynchronously
    await loop.run_in_executor(None, File_maker, filename, df)

    # Define the path to the 'plots' directory
    folder_path = './static/plots/' 

    # Remove all files in the 'plots' directory
    for filenames in os.listdir(folder_path):
        files_path = os.path.join(folder_path, filenames)
        os.remove(files_path)

    # Remove the original CSV file
    os.remove(os.path.join('./static/' + filename))

    # Return a success message with a status code of 200
    return 'message' , 200



@app.route('/pdf', methods=['GET'])
async def download_pdf():
    pdf_path = './static/example.pdf'
    if os.path.exists(pdf_path):
        return send_from_directory('./static', 'example.pdf', as_attachment=True)
    else:
        return "PDF file not found"

ch = 8
class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 8, 'Header', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

def File_maker(filename , data):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(w=0, h=20, txt=f"Name  :   {filename}", ln=1)
    pdf.set_font('Arial', '', 16)

    today = date.today()
    d = today.strftime("%B, %d, %Y")
    pdf.cell(w=30, h=ch, txt="Date: ", ln=0)
    pdf.cell(w=30, h=ch, txt=f"{d}", ln=1)

    cols_per_page = 7
    df = data
    n_cols = df.shape[1]
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(w=0, h=20, txt="Sample Data :-", ln=1)
    for col in range(min(n_cols, cols_per_page)):
        header = str(df.columns[col])
        pdf.cell(40, 10, header, 1)
    pdf.ln()
    pdf.set_font('Arial', '', 12)
    for row in range(min(df.shape[0], 5)):
        for col in range(min(n_cols, cols_per_page)):
            cell = str(df.iloc[row, col])
            pdf.cell(40, 10, cell, 1)
        pdf.ln()
    while n_cols > cols_per_page:
        pdf.add_page()
        for col in range(cols_per_page, min(n_cols, 2*cols_per_page)):
            header = str(df.columns[col])
            pdf.cell(40, 10, header, 1)
        pdf.ln()
        for row in range(min(df.shape[0], 5)):
            for col in range(cols_per_page, min(n_cols, 2*cols_per_page)):
                cell = str(df.iloc[row, col])
                pdf.cell(40, 10, cell, 1)
            pdf.ln()
        n_cols -= cols_per_page

    info1= info_section(data)
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(w=0, h=20, txt="Data Details :-", ln=1)
    pdf.set_font('Arial', '', 16)
    pdf.multi_cell(h=5.0, align='L', w=0, txt=info1, border=0)
    pdf.ln()

    pdf.set_font('Arial', 'B', 20)
    pdf.cell(w=0, h=20, txt="Data Description :-", ln=1)
    stats_df = df.describe().round(2)
    table_data = pd.DataFrame(columns=['Title', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    for col in stats_df.columns:
        title = col.title()
        count = stats_df.loc['count', col]
        mean = stats_df.loc['mean', col]
        std = stats_df.loc['std', col]
        mini = stats_df.loc['min', col]
        quart25 = stats_df.loc['25%', col]
        median = stats_df.loc['50%', col]
        quart75 = stats_df.loc['75%', col]
        maxi = stats_df.loc['max', col]   
        table_data = table_data.append({'Title': title, 'Count': count, 'Mean': mean, 'Std': std,
                                        'Min': mini, '25%': quart25, '50%': median, '75%': quart75, 'Max': maxi},
                                    ignore_index=True)
    pdf.set_font('Arial', '', 8)
    col_width = pdf.w / 9
    row_height = pdf.font_size * 2
    pdf.cell(col_width, row_height, 'Title', border=1)
    pdf.cell(col_width, row_height, 'Count', border=1)
    pdf.cell(col_width, row_height, 'Mean', border=1)
    pdf.cell(col_width, row_height, 'Std', border=1)
    pdf.cell(col_width, row_height, 'Min', border=1)
    pdf.cell(col_width, row_height, '25%', border=1)
    pdf.cell(col_width, row_height, '50%', border=1)
    pdf.cell(col_width, row_height, '75%', border=1)
    pdf.cell(col_width, row_height, 'Max', border=1)
    pdf.ln()
    for _, row in table_data.iterrows():
        pdf.cell(col_width, row_height, str(row['Title']), border=1)
        pdf.cell(col_width, row_height, str(row['Count']), border=1)
        pdf.cell(col_width, row_height, str(row['Mean']), border=1)
        pdf.cell(col_width, row_height, str(row['Std']), border=1)
        pdf.cell(col_width, row_height, str(row['Min']), border=1)
        pdf.cell(col_width, row_height, str(row['25%']), border=1)
        pdf.cell(col_width, row_height, str(row['50%']), border=1)
        pdf.cell(col_width, row_height, str(row['75%']), border=1)
        pdf.cell(col_width, row_height, str(row['Max']), border=1)
        pdf.ln()

    sns_bar_plot(data)
    scatter_plot_bool(data)
    hist_graph_plot(data)
    folder_path = "./static/plots/"
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]
    for image in image_files:
        pdf.add_page()
        pdf.image(image, x = 20, y = 20, w = 160)

    pdf.set_font('Arial', 'B', 20)
    pdf.cell(w=0, h=20, txt="Summary Plots :-", ln=1)
    over_view_plotter(data)
    pdf.add_page()
    pdf.image('./static/plots/overview.png',  x = None, y = None, w = 160, h = 0, type = 'PNG')

    pdf.output('./static/example.pdf', 'F')
    return 'Done'

def scatter_plot_bool(df):
    binary_cols = [col for col in df.columns if df[col].isin(['Yes', 'No']).all()]
    int_cols = [col for col in df.columns if df[col].isin([0, 1]).all()]

    for col in binary_cols + int_cols:
        for num_col in df.select_dtypes(include=['float64', 'int64']):
            if num_col != col:
                sns.set(rc={'figure.figsize':(11.7, 8.27), "font.size":8,"axes.titlesize":8,"axes.labelsize":5})
                sns.scatterplot(x=num_col, y=col, data=df ,)
                plt.title(f'Scatter plot of {col} vs {num_col}' , )
                plt.savefig(os.path.join("./static/plots/" + f'{col}_{num_col}_scatter.png'))
                plt.clf()

def hist_graph_plot(df):
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        for num_col in df.select_dtypes(include=['float64', 'int64']):
            if num_col != col:
                sns.set(rc={'figure.figsize':(11.7, 8.27), "font.size":8,"axes.titlesize":8,"axes.labelsize":8})
                sns.histplot(data=df, x=num_col, hue=col)
                plt.title(f'Histogram of {num_col} for {col}')
                plt.savefig(os.path.join("./static/plots/" + f'{col}_{num_col}_histogram.png'))
                plt.clf()

def info_section(data):
    buffer = io.StringIO()
    data.info(verbose=True , buf=buffer)
    s = buffer.getvalue()
    return s

def sns_bar_plot(df):
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        for col in string_cols:
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts)
            plt.title(f'{col} value counts')
            plt.xlabel('Values')
            plt.ylabel('Counts')
            plt.savefig(os.path.join("./static/plots/" + f'{col}_value_counts.png'))
            plt.clf()
    else:
        print('No columns with string datatype found in the dataframe')

def over_view_plotter(df):
    num_cols = sum([df[col].dtype == 'float64' or df[col].dtype == 'int64' for col in df.columns])
    num_rows = math.ceil(num_cols / 2)
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, num_rows * 4))
    for i, col in enumerate(df.columns):
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            avg = df[col].mean()
            row_idx = i // 2
            col_idx = i % 2
            if row_idx < len(axs) and col_idx < len(axs[row_idx]):
                axs[row_idx, col_idx].plot(df.index, df[col])
                axs[row_idx, col_idx].axhline(avg, color='red')
                axs[row_idx, col_idx].set_title(col)
                axs[row_idx, col_idx].set_xlabel('Index')
    plt.tight_layout()
    plt.savefig(os.path.join('./static/plots/' + 'overview.png'))


if __name__ == '__main__':
    app.run()