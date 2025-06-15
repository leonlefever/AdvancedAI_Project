import gradio as gr
import traceback

# === Query handler
def answer_question(qa, query):
    try:
        print(f"[INFO] Query received: {query}")
        response = qa.invoke(query)
        answer = response["result"]
        sources = "\n\n".join([doc.page_content for doc in response["source_documents"]])
        return answer, sources, "✅ Done"
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return f"❌ Error: {e}", "", "❌ Failed"

# === Gradio UI
def create_ui(qa):
    with gr.Blocks(css="footer {display: none !important}") as demo:
        with gr.Column(elem_id="main-container"):
            gr.Markdown("## 🏠 Madrid Real Estate Assistant")
            gr.Markdown("Ask about housing prices, neighborhoods, or listings in Madrid.")

            with gr.Row():
                with gr.Column(scale=4):
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
                    status_output = gr.Markdown("")

                with gr.Column(scale=5):
                    response_output = gr.Textbox(label="✅ Assistant Response", lines=10, interactive=False)

            with gr.Accordion("📄 Retrieved Listings (Sources)", open=False):
                sources_output = gr.Textbox(lines=10, interactive=False)

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

            with gr.Accordion("🏷️ Ask price based on custom property", open=False):
                sq_mt_built = gr.Slider(20, 500, value=80, step=1, label="Size (m²)")
                n_rooms = gr.Slider(1, 10, value=3, step=1, label="Number of Rooms")
                n_bathrooms = gr.Slider(1, 5, value=2, step=1, label="Number of Bathrooms")
                neighborhood_id = gr.Number(value=23, label="Neighborhood ID")
                built_year = gr.Number(value=2000, label="Year Built")

                has_lift = gr.Checkbox(label="Lift")
                has_terrace = gr.Checkbox(label="Terrace")
                has_pool = gr.Checkbox(label="Pool")
                has_parking = gr.Checkbox(label="Parking")

                generate_query_btn = gr.Button("📤 Ask LLM for Price")
                
                def construct_query_and_ask(
                    sqm, rooms, baths, hood, year, lift, terrace, pool, parking
                ):
                    features = []
                    if lift: features.append("lift")
                    if terrace: features.append("terrace")
                    if pool: features.append("pool")
                    if parking: features.append("parking")
                    features_str = ", ".join(features) if features else "no extra features"
                    
                    query = (
                        f"What is the expected price for a {sqm} m² property in neighborhood {int(hood)} "
                        f"with {rooms} rooms, {baths} bathrooms, built in {int(year)}, with {features_str}?"
                    )
                    return answer_question(qa, query)

                generate_query_btn.click(
                    fn=construct_query_and_ask,
                    inputs=[sq_mt_built, n_rooms, n_bathrooms, neighborhood_id, built_year,
                            has_lift, has_terrace, has_pool, has_parking],
                    outputs=[response_output, sources_output, status_output]
                )


    return demo
