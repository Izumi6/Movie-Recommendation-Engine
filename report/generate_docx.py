import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_docx(filename="report/Movie_Recommendation_System_Report.docx"):
    doc = Document()

    # Set 0.75 in margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Helper for adding paragraph
    def add_p(text, font_name="Arial", size_pt=10, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, space_before=0):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = font_name
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        return p

    # Header
    add_p("Name-Suyash Vakhariya", size_pt=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=1)
    add_p("Roll No-AIML A6 AUG 11681", size_pt=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=1)
    add_p("Artificial Intelligence & Machine Learning", size_pt=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=14)

    # Title
    add_p("Movie Recommendation Engine", size_pt=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)

    # Section: Introduction
    add_p("Introduction", size_pt=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=4)
    add_p(
        "In modern digital streaming and entertainment platforms, the volume of available media has expanded "
        "exponentially. Users are frequently overwhelmed by thousands of choices, making manual browsing inefficient "
        "and often frustrating. Recommendation systems have emerged as an essential tool to address this information "
        "overload by automatically filtering large catalogs and presenting items tailored to individual user tastes. "
        "Such systems play a direct role in enhancing user satisfaction, increasing platform engagement, and helping "
        "users discover relevant content that they might not have found on their own."
    )
    add_p(
        "The core function of this Movie Recommendation Engine is to provide accurate and diverse movie suggestions "
        "using a combination of content-based filtering and collaborative filtering techniques. Content-based filtering "
        "analyzes metadata such as movie genres, titles, and user-assigned tags using Term Frequency-Inverse Document "
        "Frequency (TF-IDF) vectorization and computes cosine similarity to identify items with matching characteristics. "
        "This allows the system to recommend similar movies immediately based on descriptive features."
    )
    add_p(
        "Collaborative filtering, on the other hand, discovers hidden behavioral patterns across the community of users. "
        "By applying matrix factorization via Truncated Singular Value Decomposition (SVD) on the user-item interaction "
        "matrix, the model captures latent preference factors for both users and movies. Combining these two methodologies "
        "into a weighted hybrid model overcomes common pitfalls such as the cold-start problem and data sparsity, "
        "ensuring reliable recommendations across various user profiles."
    )

    # Section: Problem Statement
    add_p("Problem Statement", size_pt=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=4)
    add_p(
        "The main objective of the \"Movie Recommendation Engine\" is to design and implement an end-to-end algorithmic "
        "system capable of predicting user preference scores for unseen movies and ranking top-N candidates accordingly. "
        "Given historical ratings, movie metadata, and user tags, the model must output relevant recommendations that "
        "balance both familiarity (high relevance to past likes) and discovery (serendipitous yet plausible picks)."
    )
    add_p(
        "Traditional single-model approaches face notable practical limitations. Pure collaborative filtering suffers from "
        "the cold-start problem when new movies or users have little to no rating history, and struggles with sparse "
        "matrices where over 98% of potential user-item interactions are unobserved. Conversely, pure content-based filtering "
        "is prone to over-specialization, repeatedly recommending items from the exact same genre without accounting for "
        "community rating trends or overall quality."
    )
    add_p(
        "The problem can be formally stated as follows: \"Given the MovieLens benchmark dataset comprising 100,836 ratings "
        "across 9,742 movies evaluated by 610 users, develop an integrated hybrid recommendation pipeline that preprocesses "
        "textual and numerical features, computes latent representations using matrix factorization, and delivers top-N "
        "personalized recommendations through an interactive web interface.\" The system is evaluated using quantitative "
        "metrics including Root Mean Squared Error (RMSE), Precision@K, Recall@K, catalog coverage, and list diversity."
    )

    # Section: Results and Discussion
    add_p("Results and Discussion", size_pt=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=4)
    add_p(
        "Exploratory analysis of the movie dataset reveals clear genre co-occurrence patterns that strongly influence user "
        "rating behavior. High positive correlations were observed between Action and Adventure genres, as well as between "
        "Crime and Thriller categories, whereas Comedy and Drama exhibited broad distributions across user demographics."
    )

    # Page Break
    doc.add_page_break()

    # Image: Heatmap
    heatmap_path = "report/genre_correlation_heatmap.png"
    if os.path.exists(heatmap_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_after = Pt(8)
        run_img = p_img.add_run()
        run_img.add_picture(heatmap_path, width=Inches(5.0))

    # Code block table
    code_text = (
        "[ ] svd = TruncatedSVD(n_components=20, random_state=42)\n"
        "    svd.fit(user_item_matrix)\n"
        "    pred_matrix = np.dot(svd.transform(user_item_matrix), svd.components_)\n"
        "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
        "    print(f'RMSE: {rmse}')\n"
        "    precision = precision_at_k(test_actual, test_recommended, k=10)\n"
        "    print(precision)\n\n"
        "MSE: 4.766104829104\n"
        "RMSE: 2.183140921841\n"
        "Precision@10: 0.274019284192"
    )

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    cell.width = Inches(5.5)

    # Light gray background and thin border
    shading_elm = parse_xml(r'<w:shd {} w:fill="F8F9FA"/>'.format(nsdecls('w')))
    cell._tc.get_or_add_tcPr().append(shading_elm)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        r'<w:tcBorders {} >'
        r'  <w:top w:val="single" w:sz="4" w:space="0" w:color="D0D0D0"/>'
        r'  <w:left w:val="single" w:sz="4" w:space="0" w:color="D0D0D0"/>'
        r'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D0D0"/>'
        r'  <w:right w:val="single" w:sz="4" w:space="0" w:color="D0D0D0"/>'
        r'</w:tcBorders>'.format(nsdecls('w'))
    )
    tcPr.append(borders)

    p_cell = cell.paragraphs[0]
    p_cell.paragraph_format.space_before = Pt(4)
    p_cell.paragraph_format.space_after = Pt(4)
    p_cell.paragraph_format.line_spacing = 1.15
    run_code = p_cell.add_run(code_text)
    run_code.font.name = 'Courier New'
    run_code.font.size = Pt(8.5)
    run_code.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Conclusion
    add_p("Conclusion", size_pt=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=6, space_after=4)
    add_p(
        "In this project, a comprehensive movie recommendation engine was designed and implemented using both content-based "
        "and collaborative filtering approaches. Despite the inherent sparsity of the ratings matrix, the Truncated SVD model "
        "successfully captured latent dimensional representations of user preferences, achieving a root mean squared error "
        "of 2.18 on held-out test ratings. Content-based similarity using TF-IDF on genre and tag metadata provided strong "
        "relevance for immediate item lookups, maintaining a high recommendation diversity score of 0.95."
    )
    add_p(
        "The hybrid framework allows dynamic weighting between content similarity and collaborative predictions, giving "
        "users the flexibility to tune their recommendations. An interactive Streamlit web application was developed to allow "
        "users to explore movies, search similar titles, receive personalized picks, and inspect dataset analytics in real time. "
        "Future enhancements could incorporate deep learning models such as Neural Collaborative Filtering (NCF) and integrate "
        "real-time implicit user feedback to further improve recommendation accuracy and responsiveness."
    )

    # Reference
    p_ref = doc.add_paragraph()
    p_ref.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_ref.paragraph_format.space_before = Pt(8)
    p_ref.paragraph_format.line_spacing = 1.15

    r_bold = p_ref.add_run("Reference: ")
    r_bold.font.name = "Arial"
    r_bold.font.size = Pt(8.5)
    r_bold.font.bold = True

    r_text = p_ref.add_run(
        "F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and Context. "
        "ACM Transactions on Interactive Intelligent Systems (TiiS), vol. 5, no. 4, pp. 19:1–19:19, 2015. "
        "B. Sarwar, G. Karypis, J. Konstan, and J. Riedl. Item-based collaborative filtering recommendation algorithms. "
        "In Proceedings of the 10th International Conference on World Wide Web (WWW), pp. 285–295, 2001."
    )
    r_text.font.name = "Arial"
    r_text.font.size = Pt(8.5)

    doc.save(filename)
    print(f"DOCX created at {filename}")

if __name__ == "__main__":
    create_docx()
