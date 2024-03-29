 // Init
 $('.image-section').hide();
 $('.loader').hide();
 $('#result').hide();
 
   
 // Upload Preview
 function readURL(input) {
     if (input.files && input.files[0]) {
         var reader = new FileReader();
         reader.onload = function (e) {
             $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
             $('#imagePreview').hide();
             $('#imagePreview').fadeIn(650);
         }
         reader.readAsDataURL(input.files[0]);
     }
 }
 $("#imageUpload").change(function () {
     $('.image-section').show();
     $('#btn-predict').show();
     $('#result').text('');
     $('#result').hide();
     readURL(this);
 });


 // Predict
 $('#btn-predict').click(function () {
     var form_data = new FormData($('#upload-file')[0]);
     console.log(form_data);
 
     // Show loading animation
     $(this).hide();
     $('.loader').show();

     // Make prediction by calling api /predict
     $.ajax({
         type: 'POST',
         url: '/predict',
         data: form_data,
         contentType: false,
         cache: false,
         processData: false,
         async: true,
         success: function (data) {
             // Get and display the result
             $('.loader').hide();
             $('#result').fadeIn(600);
             $('#result').text(' Result:  ' + data);
             console.log('Success!');
             $.ajax({
                 url: '/id/predict/' + data,
                 success: function(data){
                 var PANEL = d3.select("#sample-metadata");
                 console.log("in here");
                 // Use `.html("") to clear any existing metadata
                 PANEL.html("");
                 console.log(data);
                 data2=data[0];
                 Object.entries(data2).forEach((key) => {
                     PANEL.append("h6").text(`${key}`);
                 });
                 }
                })
         },
     });
 });

 $('#btn-predict').click(function () {
     var form_data2 = new FormData($('#upload-file')[0]);
     console.log(form_data2);
 
     // Show loading animation
     $(this).hide();
     $('.loader').show();

     // Make prediction by calling api /predict
     $.ajax({
         type: 'POST',
         url: '/predict2',
         data: form_data2,
         contentType: false,
         cache: false,
         processData: false,
         async: true,
         success: function (data) {
             // Get and display the result
             $('.loader').hide();
             $('#result2').fadeIn(600);
             $('#result2').text(' Runner-up:  ' + data);
             console.log('Success!');
         },
     });
 });