from dotenv import load_dotenv

load_dotenv()

from graph.graph import graph

if __name__ == "__main__":
    print("Hello! bvc")
    graph.get_graph().draw_mermaid_png(output_file_path="./graph.png")

    while True:
        query = input("How can I help you?: ")
        res = graph.invoke(input={"question": query})
        # print("result: ", res)

        # Safely access 'generation'
        generation = res["generation"]
        print(generation)
