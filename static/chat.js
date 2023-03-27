window.onload = function () {
document
  .getElementById("chat-form")
  .addEventListener("submit", function (event) {
    // Prevent the form from submitting and refreshing the page
    event.preventDefault();

    let userInput = document.getElementById("user-input").value;
    let url = `/chat?user_input=${encodeURIComponent(userInput)}`;

    fetch(url)
      .then((response) => response.json())
      .then((data) => {
        let content = data.content;
        let resultDiv = document.getElementById("result");
        resultDiv.innerHTML = content;
      })
      .catch((error) => {
        console.error("Error fetching GPT-3 response:", error);
      });
  });
};