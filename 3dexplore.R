library(plotly)
library(shiny)
library(htmlwidgets)

initDF <- data.frame(x = 1:10, y = 1:10)

ui <- fluidPage(
  plotlyOutput("myPlot"),
  verbatimTextOutput("click")
)

server <- function(input, output, session) {
  
  js <- "
    function(el, x, inputName){
      var id = el.getAttribute('id');
      var gd = document.getElementById(id);
      var d3 = Plotly.d3;
      Plotly.update(id).then(attach);
        function attach() {
          gd.addEventListener('click', function(evt) {
            var xaxis = gd._fullLayout.xaxis;
            var yaxis = gd._fullLayout.yaxis;
            var bb = evt.target.getBoundingClientRect();
            var x = xaxis.p2d(evt.clientX - bb.left);
            var y = yaxis.p2d(evt.clientY - bb.top);
            var coordinates = [x, y];
            Shiny.setInputValue(inputName, coordinates);
          });
        };
  }
  "
  
  clickposition_history <- reactiveVal(initDF)
  
  observeEvent(input$clickposition, {
    clickposition_history(rbind(clickposition_history(), input$clickposition))
  })
  
  output$myPlot <- renderPlotly({
    plot_ly(initDF, x = ~x, y = ~y, type = "scatter", mode = "markers") %>%
      onRender(js, data = "clickposition")
  })
  
  myPlotProxy <- plotlyProxy("myPlot", session)
  
  observe({
    plotlyProxyInvoke(myPlotProxy, "restyle", list(x = list(clickposition_history()$x), y = list(clickposition_history()$y)))
  })
  
  output$click <- renderPrint({
    clickposition_history()
  })
}

shinyApp(ui, server)