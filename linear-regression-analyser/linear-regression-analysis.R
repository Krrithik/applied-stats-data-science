library(shiny)
library(DT)
library(MASS)
library(lmtest)
library(car)
library(ggplot2)
library(shinythemes)
library(nortest)

# Check for ISLR2, load if available (contains datasets like College, Auto)
if(require(ISLR2)) {
  library(ISLR2)
}

# ---------- HELPERS ----------

# Generate formulas for all subsets (limited to prevent crashes)
all_subset_formulas <- function(response, preds) {
  if (length(preds) == 0) return(list())
  out <- list()
  k <- 1
  for (m in 1:length(preds)) {
    # Safety limit: if >10 vars, don't try combinations larger than 4
    if(m > 4 && length(preds) > 10) next 
    combs <- combn(preds, m, simplify = FALSE)
    for (cc in combs) {
      out[[k]] <- as.formula(paste(response, "~", paste(cc, collapse = " + ")))
      k <- k + 1
    }
  }
  out
}

mse <- function(y, yhat) mean((y - yhat)^2, na.rm = TRUE)

fmt_equation <- function(model) {
  co <- coef(model)
  nm <- names(co)
  eq <- paste0("ŷ = ", round(co[1], 4))
  if (length(co) > 1) {
    for (i in 2:length(co)) {
      s <- ifelse(co[i] >= 0, " + ", " - ")
      eq <- paste0(eq, s, abs(round(co[i], 4)), "·", nm[i])
    }
  }
  eq
}

safe <- function(x) tryCatch(x, error = function(e) NA)

# Power transform helper
power_transform <- function(v, p) {
  v <- as.numeric(v)
  if (abs(p) < 1e-8) {
    shift <- if (any(v <= 0, na.rm = TRUE)) abs(min(v, na.rm = TRUE)) + 1 else 0
    return(log(v + shift))
  }
  if (any(v < 0, na.rm = TRUE) && abs(p - round(p)) > 1e-8) {
    shift <- abs(min(v, na.rm = TRUE)) + 1
    v <- v + shift
  }
  v^p
}

# ---------- UI ----------
ui <- fluidPage(theme = shinythemes::shinytheme("cerulean"),
                titlePanel("Linear Regression Analyser"),
                
                sidebarLayout(
                  sidebarPanel(
                    h4("1. Data Source"),
                    radioButtons("src", "Source",
                                 c("Upload CSV" = "upload",
                                   "Built-in / ISLR datasets" = "builtin")),
                    conditionalPanel(
                      condition = "input.src=='upload'",
                      fileInput("file", "Upload CSV", accept = ".csv")
                    ),
                    conditionalPanel(
                      condition = "input.src=='builtin'",
                      selectInput("ds", "Dataset",
                                  c("mtcars", "iris", "Auto (ISLR)", "Boston (ISLR)", 
                                    "College (ISLR)", "Carseats (ISLR)", "Wage (ISLR)", "Hitters (ISLR)"))
                    ),
                    
                    tags$hr(),
                    h4("2. Variables"),
                    uiOutput("var_ui"),
                    checkboxInput("all_subsets", "Use all subset regression", TRUE),
                    
                    tags$hr(),
                    h4("3. Train/Test Split"),
                    numericInput("seed", "Seed", 123),
                    sliderInput("train_prop", "Training proportion", 0.5, 0.9, 0.7, step = 0.05),
                    actionButton("fit", "Split and Fit Models", class = "btn-primary"),
                    
                    tags$hr(),
                    h4("4. Model Selection"),
                    selectInput("crit", "Select best model by",
                                c("Test MSE" = "mse",
                                  "AIC (train)" = "aic",
                                  "BIC (train)" = "bic")),
                    
                    tags$hr(),
                    h4("Settings"),
                    selectInput("alpha", "Significance level α",
                                c("0.10" = 0.10, "0.05" = 0.05, "0.01" = 0.01),
                                selected = 0.05)
                  ),
                  
                  mainPanel(
                    tabsetPanel(
                      tabPanel("Data", DTOutput("tbl")),
                      
                      tabPanel("Models",
                               h4("All Candidate Models"),
                               DTOutput("metrics"),
                               tags$hr(),
                               h4("Best Model Selected"),
                               verbatimTextOutput("summary"),
                               h5("Model Performance Metrics:"),
                               verbatimTextOutput("best_aic_bic"),
                               tags$hr(),
                               h4("Fitted Equation"),
                               verbatimTextOutput("equation"),
                               tags$hr(),
                               h4("Coefficients & P-Values"),
                               DTOutput("coef_tbl")
                      ),
                      
                      tabPanel("Step 4: Assumption Tests",
                               h4("Hypothesis Tests on Residuals (Best Model)"),
                               helpText("Checking the assumptions of the currently selected 'Best Model'."),
                               tags$strong("1. Normality of Residuals (Shapiro-Wilk Test)"),
                               verbatimTextOutput("test_shapiro"),
                               helpText("Null Hypothesis: Residuals are normally distributed. (p < 0.05 implies violation)"),
                               tags$hr(),
                               tags$strong("2. Homoscedasticity (Breusch-Pagan Test)"),
                               verbatimTextOutput("test_bp"),
                               helpText("Null Hypothesis: Variance is constant. (p < 0.05 implies heteroscedasticity)"),
                               tags$hr(),
                               tags$strong("3. Multicollinearity (VIF)"),
                               verbatimTextOutput("test_vif")
                      ),
                      
                      tabPanel("Diagnostics (Plots)",
                               h4("Residual Visualizations"),
                               fluidRow(
                                 column(6, plotOutput("res_hist", height = 300)),
                                 column(6, plotOutput("res_box", height = 300))
                               ),
                               fluidRow(
                                 column(6, plotOutput("res_vs_fitted", height = 300)),
                                 column(6, plotOutput("res_qq", height = 300))
                               ),
                               tags$hr(),
                               h4("Assumption Checks (Summary Table)"),
                               DTOutput("assump_tbl")
                      ),
                      
                      tabPanel("Prediction",
                               h4("Predict using Best Model"),
                               uiOutput("pred_ui"),
                               actionButton("pred", "Calculate Prediction", class = "btn-success"),
                               tags$br(), tags$br(),
                               verbatimTextOutput("pred_out")
                      ),
                      
                      tabPanel("EDA: Step 1 (Y)",
                               sliderInput("y_power", "Power transform for Y (0 means log)", min = -3, max = 3, value = 1, step = 0.05),
                               helpText("Visualizations styled after Daniel Rivera's Didactic App"),
                               fluidRow(
                                 column(4, plotOutput("y_hist", height = 350)),
                                 column(4, plotOutput("y_box",  height = 350)),
                                 column(4, plotOutput("y_qq",   height = 350))
                               )
                      ),
                      
                      tabPanel("EDA: Step 2 (Y vs X)",
                               selectInput("eda_x", "Choose one predictor X to explore", choices = NULL),
                               sliderInput("x_power", "Power transform for X (0 means log)", min = -3, max = 3, value = 1, step = 0.05),
                               checkboxInput("apply_y_power_in_step2", "Also apply Y power transform here", TRUE),
                               verbatimTextOutput("eda_cor"),
                               plotOutput("yx_scatter", height = 400)
                      )
                    )
                  )
                )
)

# ---------- SERVER ----------
server <- function(input, output, session) {
  
  # 1. Unified Data Loading
  dat <- reactive({
    if (input$src == "upload") {
      req(input$file)
      read.csv(input$file$datapath)
    } else {
      switch(input$ds,
             "mtcars" = mtcars,
             "iris" = iris,
             "Auto (ISLR)" = if(exists("Auto")) Auto else NULL,
             "Boston (ISLR)" = if(exists("Boston")) Boston else NULL,
             "College (ISLR)" = if(exists("College")) College else NULL,
             "Carseats (ISLR)" = if(exists("Carseats")) Carseats else NULL,
             "Wage (ISLR)" = if(exists("Wage")) Wage else NULL,
             "Hitters (ISLR)" = if(exists("Hitters")) na.omit(Hitters) else NULL)
    }
  })
  
  output$tbl <- renderDT({
    req(dat())
    datatable(head(dat(), 50), options = list(scrollX = TRUE))
  })
  
  # 2. Variable Selection
  output$var_ui <- renderUI({
    d <- dat()
    req(d)
    vars <- names(d)
    # Default to numeric for X if possible
    nums <- names(d)[sapply(d, is.numeric)]
    tagList(
      selectInput("y", "Response (Y)", choices = vars),
      selectizeInput("x", "Predictors (X)", choices = nums, multiple = TRUE)
    )
  })
  
  # EDA: Update X choice based on data
  observe({
    d <- dat()
    req(input$y)
    updateSelectInput(session, "eda_x", choices = setdiff(names(d), input$y))
  })
  
  # 3. EDA Logic
  y_transformed <- reactive({
    d <- dat()
    req(input$y)
    y <- d[[input$y]]
    y <- y[!is.na(y)]
    power_transform(y, input$y_power)
  })
  
  output$y_hist <- renderPlot({
    y <- y_transformed()
    df <- data.frame(val = y)
    ggplot(df, aes(x = val)) +
      geom_histogram(bins = nclass.Sturges(y), color = "white",
                     fill = "seagreen1", aes(y = ..density..), lwd = 0.8) +
      geom_density(color = "seagreen4", alpha = 0.3, fill = "seagreen4", lty = 1) +
      labs(title = paste(input$y, "\n histogram"), x = input$y, y = "Density") +
      theme_minimal()
  })
  
  output$y_box <- renderPlot({
    y <- y_transformed()
    df <- data.frame(val = y)
    ggplot(df, aes(x = 0, y = val)) +
      geom_boxplot(color = "black", fill = "skyblue", alpha = 0.5) +
      stat_summary(fun = mean, colour = "darkred", geom = "point", shape = 18, size = 3) +
      labs(title = paste(input$y, "\n boxplot"), x = "", y = input$y) +
      theme_minimal()
  })
  
  output$y_qq <- renderPlot({
    y <- y_transformed()
    car::qqPlot(y, col = "coral", pch = 16, id = FALSE, lwd = 1.9, 
                col.lines = "black", grid = FALSE, 
                main = paste(input$y, "\n Q-Q plot"), xlab = "Normal quantiles", ylab = input$y)
  })
  
  output$yx_scatter <- renderPlot({
    d <- dat()
    req(input$y, input$eda_x)
    y <- d[[input$y]]
    x <- d[[input$eda_x]]
    
    if(isTRUE(input$apply_y_power_in_step2)) y <- power_transform(y, input$y_power)
    x <- power_transform(x, input$x_power)
    
    df <- data.frame(x = x, y = y)
    ggplot(df, aes(x = x, y = y)) +
      geom_point(shape = 18, color = "blue", size = 3) +
      geom_smooth(method = lm, linetype = "dashed", color = "black", fill = "seagreen3") +
      labs(title = paste("\n", input$y, "vs", input$eda_x, "\n"), 
           x = input$eda_x, y = input$y) +
      theme_minimal()
  })
  
  output$eda_cor <- renderPrint({
    d <- dat()
    req(input$y, input$eda_x)
    y <- d[[input$y]]
    x <- d[[input$eda_x]]
    if(isTRUE(input$apply_y_power_in_step2)) y <- power_transform(y, input$y_power)
    x <- power_transform(x, input$x_power)
    cat("Correlation:", cor(x, y, use="complete.obs"))
  })
  
  # 4. Model Fitting Logic
  rv <- reactiveValues(best = NULL, metrics = NULL, models = NULL, train = NULL, test = NULL)
  
  observeEvent(input$fit, {
    req(input$y, input$x)
    validate(need(length(input$x) >= 1, "Select at least one predictor (X)."))
    
    d <- dat()
    xvars <- setdiff(input$x, input$y)
    validate(need(length(xvars) >= 1, "Pick at least one predictor X that is not the response Y."))
    
    # Filter data
    keep <- unique(c(input$y, xvars))
    d <- d[complete.cases(d[, keep, drop = FALSE]), keep, drop = FALSE]
    
    set.seed(input$seed)
    idx <- sample(nrow(d))
    ntr <- floor(input$train_prop * nrow(d))
    tr <- d[idx[1:ntr], , drop = FALSE]
    te <- d[idx[(ntr + 1):nrow(d)], , drop = FALSE]
    
    # Generate formulas
    forms <- if (isTRUE(input$all_subsets)) {
      all_subset_formulas(input$y, xvars)
    } else {
      list(as.formula(paste(input$y, "~", paste(xvars, collapse = "+"))))
    }
    
    mods <- lapply(forms, lm, data = tr)
    
    # Calculate metrics (AIC/BIC included here)
    mets <- do.call(rbind, lapply(seq_along(mods), function(i) {
      m <- mods[[i]]
      ytrue <- te[[input$y]]
      yhat <- predict(m, te)
      data.frame(
        id = i,
        formula = deparse(formula(m)),
        adjR2 = summary(m)$adj.r.squared,
        mse = mse(ytrue, yhat),
        aic = AIC(m),
        bic = BIC(m),
        stringsAsFactors = FALSE
      )
    }))
    
    ord <- switch(input$crit,
                  mse = order(mets$mse, -mets$adjR2),
                  aic = order(mets$aic, mets$mse),
                  bic = order(mets$bic, mets$mse))
    
    rv$train <- tr
    rv$test <- te
    rv$models <- mods
    rv$metrics <- mets
    rv$best <- mods[[mets$id[ord[1]]]]
  })
  
  output$metrics <- renderDT({
    req(rv$metrics)
    # Format the table nicely
    datatable(rv$metrics, options = list(scrollX = TRUE, pageLength = 5), rownames = FALSE) %>%
      formatRound(columns = c("adjR2", "mse", "aic", "bic"), digits = 4)
  })
  
  output$summary <- renderPrint({ req(rv$best); summary(rv$best) })
  
  # Explicit AIC/BIC output
  output$best_aic_bic <- renderPrint({
    req(rv$best)
    cat("AIC:", AIC(rv$best), "\n")
    cat("BIC:", BIC(rv$best))
  })
  
  output$equation <- renderPrint({ req(rv$best); cat(fmt_equation(rv$best)) })
  
  # FIXED: Coefficient Table Construction
  output$coef_tbl <- renderDT({
    req(rv$best)
    s <- summary(rv$best)$coefficients
    # Correct way to make dataframe from summary matrix
    df <- as.data.frame(s)
    df$Term <- rownames(s)
    # Reorder columns: Term first
    df <- df[, c("Term", "Estimate", "Std. Error", "t value", "Pr(>|t|)")]
    names(df) <- c("Term", "Estimate", "Std. Error", "t value", "p-value")
    
    datatable(df, options = list(scrollX = TRUE, pageLength = 10), rownames = FALSE) %>%
      formatRound(columns = c("Estimate", "Std. Error", "t value"), digits = 4) %>%
      formatSignif(columns = c("p-value"), digits = 4)
  })
  
  # --- DIAGNOSTICS ---
  output$res_hist <- renderPlot({
    req(rv$best)
    res <- residuals(rv$best)
    ggplot(data.frame(res), aes(x=res)) + geom_histogram(bins=15, fill="seagreen1", color="white") +
      geom_density(fill="seagreen4", alpha=0.3) + theme_minimal() + labs(title="Residuals Histogram")
  })
  
  output$res_box <- renderPlot({
    req(rv$best)
    res <- residuals(rv$best)
    ggplot(data.frame(res), aes(x=0, y=res)) + geom_boxplot(fill="skyblue") + 
      labs(title="Residuals Boxplot", x="") + theme_minimal()
  })
  
  output$res_vs_fitted <- renderPlot({
    req(rv$best)
    df <- data.frame(fitted=fitted(rv$best), resid=residuals(rv$best))
    ggplot(df, aes(x=fitted, y=resid)) + geom_point(color="blue") + geom_smooth(method="lm", se=FALSE, color="red") +
      theme_minimal() + labs(title="Residuals vs Fitted")
  })
  
  output$res_qq <- renderPlot({
    req(rv$best)
    car::qqPlot(residuals(rv$best), col="coral", pch=16, grid=FALSE, main="Q-Q Plot")
  })
  
  output$assump_tbl <- renderDT({
    req(rv$best)
    a <- as.numeric(input$alpha)
    m <- rv$best
    s <- summary(m)
    res <- residuals(m)
    
    # FIXED: Robust Shapiro for large N
    p_shap <- if(length(res) > 5000) safe(shapiro.test(sample(res, 5000))$p.value) else safe(shapiro.test(res)$p.value)
    p_bp   <- safe(bptest(m)$p.value)
    p_dw   <- safe(dwtest(m)$p.value)
    
    fstat <- s$fstatistic
    p_f <- if(!is.null(fstat)) safe(pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE)) else NA
    
    df <- data.frame(
      Check = c("Normality (Shapiro-Wilk)", "Homoscedasticity (Breusch-Pagan)", 
                "Independence (Durbin-Watson)", "Model Significance (F-test)"),
      `p-value` = c(p_shap, p_bp, p_dw, p_f),
      Decision = c(
        ifelse(is.na(p_shap), "Unavailable", ifelse(p_shap >= a, "PASS", "FLAG")),
        ifelse(is.na(p_bp),   "Unavailable", ifelse(p_bp   >= a, "PASS", "FLAG")),
        ifelse(is.na(p_dw),   "Unavailable", ifelse(p_dw   >= a, "PASS", "FLAG")),
        ifelse(is.na(p_f),    "Unavailable", ifelse(p_f    <  a, "SIGNIFICANT", "Not significant"))
      ),
      stringsAsFactors = FALSE
    )
    datatable(df, options = list(scrollX = TRUE, paging = FALSE), rownames = FALSE) %>%
      formatSignif(columns = "p.value", digits = 4)
  })
  
  output$test_shapiro <- renderPrint({
    req(rv$best)
    res <- residuals(rv$best)
    if (length(res) > 5000) shapiro.test(sample(res, 5000)) else shapiro.test(res)
  })
  
  output$test_bp <- renderPrint({ req(rv$best); lmtest::bptest(rv$best) })
  output$test_vif <- renderPrint({ 
    req(rv$best)
    if(length(coef(rv$best)) < 3) cat("VIF requires >= 2 predictors") else car::vif(rv$best) 
  })
  
  # 6. PREDICTION (FIXED LOGIC)
  output$pred_ui <- renderUI({
    req(rv$best, rv$train)
    xs <- all.vars(delete.response(terms(rv$best)))
    
    tagList(lapply(xs, function(x) {
      # Check if x is factor in training data
      if (!is.null(rv$train[[x]]) && (is.factor(rv$train[[x]]) || is.character(rv$train[[x]]))) {
        selectInput(paste0("new_", x), x, choices = unique(rv$train[[x]]))
      } else {
        numericInput(paste0("new_", x), x, value = 0)
      }
    }))
  })
  
  # Logic: Use eventReactive to calculate prediction only when button is clicked
  prediction_val <- eventReactive(input$pred, {
    req(rv$best, rv$train)
    xs <- all.vars(delete.response(terms(rv$best)))
    
    # Construct newdata df
    newdata <- data.frame(matrix(ncol = length(xs), nrow = 1))
    names(newdata) <- xs
    
    for (x in xs) {
      input_id <- paste0("new_", x)
      val <- input[[input_id]]
      
      # Handle potential missing input during render
      if(is.null(val)) return(NULL)
      
      if (!is.null(rv$train[[x]]) && (is.factor(rv$train[[x]]) || is.character(rv$train[[x]]))) {
        newdata[[x]] <- factor(val, levels = levels(as.factor(rv$train[[x]])))
      } else {
        newdata[[x]] <- as.numeric(val)
      }
    }
    
    predict(rv$best, newdata)
  })
  
  output$pred_out <- renderPrint({
    req(prediction_val())
    cat("Predicted Y value:\n")
    print(prediction_val())
  })
}

shinyApp(ui, server)