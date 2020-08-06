function tabHighlighting() {
  /**
   * Highlights the current location in the navbar.
   */

  $(function () {
    $('nav').children('a').each(function () {
      if ($(this).prop('href') == window.location.href) {
        $(this).addClass('active');
      }
    });
  });
  $(function(){
    $(window).scroll(function(){
      var winTop = $(window).scrollTop();
      if(winTop >= 30){
        $("body").addClass("sticky-header");
      }else{
        $("body").removeClass("sticky-header");
      }//if-else
    });//win func.
  });//ready func.
}

function toggleMaxTextAlert() {
  /**
   * Displays a warning message if the text in the main field is at capacity.
   * When text is reduced to below capacity, the message is removed.
   * @type {HTMLElement}
   */
  const textfield = document.getElementById('text');
  textfield.oninput = function () {
    if (this.value.length >= this.maxLength) {
      document.getElementById('text-too-long').style.display = 'block';
    } else {
      document.getElementById('text-too-long').style.display = 'none';
    }
  };
}

function setSliderDisplayValue() {
  /**
   * Sets the display value in accordance with the position of the slider.
   * @type {HTMLElement}
   */
  const slider = document.getElementById('percent');
  const val = document.getElementById("percent-value");
  const setSlider = function() {
    val.innerHTML = 'Condense to %: ' + slider.value;
  };
  setSlider();
  slider.oninput = setSlider;
  // handles BFCache ("back" button on browser)
  $(window).bind('pageshow', function() {
    setSlider();
  });
}

function setLoadingButton() {
  /**
   * Changes the "condense" button to a "loading" element once clicked.
   */
  $(function(){
   $("#data").submit(function(){
     $( "#submit" ).replaceWith("<button class=\"btn btn-lg\" type=\"button\" disabled>" +
       "<span class=\"spinner-border spinner-border-sm\" role=\"status\" aria-hidden=\"true\">" +
       "</span>Loading...</button>");
   });
  });
}

function uploadFileToTextArea() {
  /**
   * When a file is uploaded, populates the text area with the text of the file.
   */
  $(function() {
    $('input[type="file"]').change(function(e){
       const fileName = e.target.files[0].name;
       const tmppath = URL.createObjectURL(e.target.files[0]);
       $.get(tmppath, function(data) {
         const textfield = document.getElementById('text');
         textfield.value = data.substring(0, Math.min(data.length, textfield.maxLength));
         textfield.dispatchEvent(new Event('input'));
       }, 'text');
     });
  });
}
