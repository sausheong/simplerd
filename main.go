package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type Prompt struct {
	Input   string `json:"input"`
	Setting string `json:"setting"`
}

var simplerPrompt map[string]string
var preamble = "A Lexile measure is defined as the numeric representation of an individual's reading ability or a text's readability (or difficulty), followed by an 'L' (Lexile). Lexile measures range from below 200L for beginning readers and text to 2000L for advanced readers. "
var handlers map[string]func(Prompt, http.ResponseWriter)

// initialise to load environment variable from .env file
func init() {
	err := godotenv.Load()
	if err != nil {
		log.Println("Error loading .env file:", err)
	}
	simplerPrompt = make(map[string]string)
	simplerPrompt["L1"] = "Rewrite the given text to Lexile text measure of 190L such that a student between 6-8 years old and in grade 1-2 can understand"
	simplerPrompt["L2"] = "Rewrite the given text to Lexile text measure of 520L such that a student between 8-10 years old and in grade 3-4 can understand."
	simplerPrompt["L3"] = "Rewrite the given text to Lexile text measure of 830L such that a student between 10-12 years old and in grade 5-6 can understand."
	simplerPrompt["L4"] = "Rewrite the given text to Lexile text measure of 970L such that a student between 12-14 years old and in grade 7-8 can understand."
	simplerPrompt["L5"] = "Rewrite the given text to Lexile text measure of 1150L such that a student between 14-16 years old and in grade 9-10 can understand."
	simplerPrompt["L6"] = "Rewrite the given text to Lexile text measure of 1185L such that a student between 16-18 years old and in grade 11-12 can understand."

	handlers = make(map[string]func(Prompt, http.ResponseWriter))
	handlers["gpt"] = gpt
	handlers["gemini"] = gemini

}

func main() {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Post("/call/{llm}", call)
	http.ListenAndServe(":"+os.Getenv("PORT"), r)
}

// call the LLM and return the response
func call(w http.ResponseWriter, r *http.Request) {
	prompt := Prompt{}
	err := json.NewDecoder(r.Body).Decode(&prompt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	llm := chi.URLParam(r, "llm")
	handlers[llm](prompt, w)
}

// using OpenAI GPT models
func gpt(prompt Prompt, w http.ResponseWriter) {
	llm, err := openai.NewChat(openai.WithModel("gpt-3.5-turbo"))
	if err != nil {
		log.Println("Cannot create openAI LLM:", err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Add("mime-type", "text/event-stream")
	f := w.(http.Flusher)
	llm.Call(context.Background(), []schema.ChatMessage{
		schema.SystemChatMessage{
			Content: preamble + simplerPrompt[prompt.Setting],
		},
		schema.HumanChatMessage{Content: prompt.Input},
	}, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		w.Write(chunk)
		f.Flush()
		return nil
	}),
	)
}

// using Google Gemini models
func gemini(prompt Prompt, w http.ResponseWriter) {
	client, err := genai.NewClient(context.Background(),
		option.WithAPIKey(os.Getenv("GOOGLEAI_API_KEY")))
	if err != nil {
		log.Println("cannot create Gemini client:", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer client.Close()

	gemini := client.GenerativeModel("gemini-pro")
	iter := gemini.GenerateContentStream(context.Background(),
		genai.Text(preamble+simplerPrompt[prompt.Setting]),
		genai.Text(prompt.Input))
	w.Header().Add("mime-type", "text/event-stream")
	f := w.(http.Flusher)
	for {
		resp, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			log.Println("cannot generate content:", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		for _, cand := range resp.Candidates {
			if cand.Content != nil {
				for _, part := range cand.Content.Parts {
					if part != nil {
						w.Write([]byte(part.(genai.Text)))
						f.Flush()
					}
				}
			}
		}
	}
}
