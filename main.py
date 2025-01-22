from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__=="__main__":
    print("Hello! bvc")
    app.get_graph().draw_mermaid_png(output_file_path="./graph.png")
    res = app.invoke(input={"question": "what is ayushman bharat"})
    print(res["generation"])