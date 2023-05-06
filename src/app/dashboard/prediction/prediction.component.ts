import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { FlaskService } from 'src/app/services/flask.service';
import Swal from 'sweetalert2';
@Component({
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.css'],
})
export class PredictionComponent implements OnInit {
  selectedCountry: string;
  date: number;
  countries: string[] = []; // Example list of countries
  selectedOption:string;
  constructor(private router:Router,private flask: FlaskService) {}

  ngOnInit(): void {

    if ((localStorage.getItem('user')==null) )
    {

      alert(' FORBIDDEN ACCESS  !!');
      this.router.navigate(['/']);

   }




    this.ListCountries();
  }

  ListCountries() {
    this.flask.getcounty().subscribe(
      (data) => {
        this.countries = data;
        console.log(this.countries);
      },
      (err) => {
        console.log(err);
      }
    );
  }

  submitForm() {
    let prediction: any = {
      Country: this.selectedCountry,
      Target_Date: this.date,
    };
    this.flask.predict(prediction).subscribe(
      (data) => {
        console.log(data);
        alert(data['predicted_emissions']);

        this.selectedCountry = '';
        this.date = 2023;
      },

      (err) => {
        console.log(err);
      }
    );
  }
  submitForm1(){
    let prediction: any = {
      country: this.selectedCountry,
      date: this.date,
    };
    this.flask.predictfreshwater(prediction).subscribe(
      (data) => {
        console.log(data);
 
      
        Swal.fire({
          title: 'Prediction!',
          text: 'our prediction of freshwater:'+data['prediction'],
          icon: 'success',
          confirmButtonColor: '#3085d6',
          confirmButtonText: 'OK'
        });
        this.selectedCountry = '';
        this.date = 2023;
      },

      (err) => {
        console.log(err);
      }
    );
  }

  submitForm2(){
    let prediction: any = {
      country: this.selectedCountry,
      date: this.date,
    };
    this.flask.predictfreshwater(prediction).subscribe(
      (data) => {
        console.log(data);
 
      
        Swal.fire({
          title: 'Poverty!',
          text: 'our Predection of Poverty:'+data['prediction'],
          icon: 'success',
          confirmButtonColor: '#3085d6',
          confirmButtonText: 'OK'
        });
        this.selectedCountry = '';
        this.date = 2023;
      },

      (err) => {
        console.log(err);
      }
    );
  }

  submitForm3(){
    let prediction: any = {
      country: this.selectedCountry,
      date: this.date,
    };
    this.flask.predectionhunger(prediction).subscribe(
      (data) => {
        console.log(data);
 
      
        Swal.fire({
          title: 'Poverty!',
          text: 'our Predection of Hunger:'+data['prediction'],
          icon: 'success',
          confirmButtonColor: '#3085d6',
          confirmButtonText: 'OK'
        });
        this.selectedCountry = '';
        this.date = 2023;
      },

      (err) => {
        console.log(err);
      }
    );
  }
  // You can now use the selectedCountry and value variables in your logic
}
