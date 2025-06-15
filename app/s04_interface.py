import gradio as gr
from gradio.themes.utils import fonts
import traceback

# Query handler
def answer_question(qa, query):
    try:
        print(f"[INFO] Query received: {query}")
        response = qa.invoke(query)
        answer = response["result"]
        sources = "\n\n".join([doc.page_content for doc in response["source_documents"]])
        return answer, sources, ""
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return f"❌ Error: {e}", "", "❌ Failed"

# Custom dark theme
dark_theme = gr.themes.Soft(
    font=fonts.GoogleFont("Roboto"),
).set(
    body_background_fill="#1c1c1c",
    body_text_color="#f1f1f1",
    button_primary_background_fill="#00897b",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#424242",
    button_secondary_text_color="#f1f1f1",
    input_background_fill="#2c2c2c",
    input_border_color="#555555"
)

# UI
def create_ui(qa):
    with gr.Blocks(theme=dark_theme) as demo:
        with gr.Column():
            gr.Markdown("## 🏠 Madrid Real Estate Assistant")
            gr.Markdown("Ask about housing prices, neighborhoods, or listings in Madrid.")

            with gr.Row():
                # Left: Chat functionality
                with gr.Column(scale=6):
                    response_output = gr.Textbox(label="✅ Assistant Response", lines=10, interactive=False)

                    with gr.Accordion("📄 Retrieved Listings (Sources)", open=False):
                        sources_output = gr.Textbox(lines=10, interactive=False)

                    status_output = gr.Markdown("")

                    query_input = gr.Textbox(
                        label="💬 Your Question",
                        placeholder="e.g. What’s the average price in Salamanca?",
                        lines=2,
                    )

                    gr.Examples(
                        examples=[
                            "What are the most expensive neighborhoods in Madrid?",
                            "Why are flats in Salamanca so expensive?",
                            "What influences the price of listings in Chamberí?",
                            "Are there affordable homes with terraces in Chamartín?",
                            "How much do homes with 2 bedrooms cost in Retiro?"
                        ],
                        inputs=query_input,
                    )

                    submit_btn = gr.Button("🔍 Ask")
                    clear_btn = gr.Button("🗑️ Clear")

                # Right: Property price prediction
                with gr.Column(scale=5):
                    gr.Markdown("### 🧮 Predict Listing Price")

                    size_min = gr.Slider(20, 500, value=60, step=1, label="Min Size (m²)")
                    size_max = gr.Slider(20, 500, value=100, step=1, label="Max Size (m²)")

                    min_rooms = gr.Dropdown(
                        choices=[str(i) for i in range(1, 11)],
                        value="3",
                        label="Min Number of Rooms"
                    )

                    min_bathrooms = gr.Dropdown(
                        choices=[str(i) for i in range(1, 6)],
                        value="2",
                        label="Min Number of Bathrooms"
                    )

                    neighborhood_dropdown = gr.Dropdown(
                        label="Neighborhood",
                        choices=[
                            "Salamanca", "Chamartín", "Retiro", "Chamberí", "Centro",
                            "Moncloa-Aravaca", "Arganzuela", "Tetuán", "Latina", "Ciudad Lineal"
                        ],
                        value="Salamanca"
                    )

                    with gr.Row():
                        has_lift = gr.Checkbox(label="Lift")
                        has_terrace = gr.Checkbox(label="Terrace")
                        has_pool = gr.Checkbox(label="Pool")
                        has_parking = gr.Checkbox(label="Parking")

                    generate_query_btn = gr.Button("📤 Ask LLM for Price")

            # Button handlers
            def wrapped_answer(query):
                return answer_question(qa, query)

            submit_btn.click(
                fn=wrapped_answer,
                inputs=[query_input],
                outputs=[response_output, sources_output, status_output]
            )

            clear_btn.click(
                fn=lambda: ("", "", "", ""),
                inputs=[],
                outputs=[query_input, response_output, sources_output, status_output]
            )

            def construct_query_and_ask(
                size_min_val, size_max_val, min_rooms_val, min_baths_val,
                neighborhood, lift, terrace, pool, parking
            ):
                features = []
                if lift: features.append("lift")
                if terrace: features.append("terrace")
                if pool: features.append("pool")
                if parking: features.append("parking")
                features_str = ", ".join(features) if features else "no extra features"

                query = (
                    f"What is the expected price for a property between {size_min_val} m² and {size_max_val} m² "
                    f"in {neighborhood}, with at least {int(min_rooms_val)} rooms and {int(min_baths_val)} bathrooms, "
                    f"with {features_str}?"
                )

                return answer_question(qa, query)

            generate_query_btn.click(
                fn=construct_query_and_ask,
                inputs=[
                    size_min, size_max, min_rooms, min_bathrooms, neighborhood_dropdown,
                    has_lift, has_terrace, has_pool, has_parking
                ],
                outputs=[response_output, sources_output, status_output]
            )

    return demo