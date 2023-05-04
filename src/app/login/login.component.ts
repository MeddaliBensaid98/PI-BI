
import { Component } from '@angular/core';
import { AuthService } from '../services/auth.service';
import { OnInit } from '@angular/core';
import {   Router } from '@angular/router';
import Swal from 'sweetalert2';
@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  username: string;
  password: string;
  cin: number;
  username2: string;
  password2: string;
  repassword2: string;
  tel: number;
  bd: Date;
  email: string;
  nom: string;
  surname: string;
  userByUsername: any;
data:any;
ok=true;constructor(private authService: AuthService,   private router: Router) {}

ngOnInit(): void {}
onSubmit() {
  let user: any = { username: this.username, password: this.password };

  this.authService.login(user).subscribe(
    (rep) => {
      if (rep.ok) {
    
        this.data = rep.body;
        localStorage.setItem("user",  JSON.stringify( this.data))
console.log( this.data)
window.location.href = 'http://localhost:4200/d';


      }  
     
      this.username = '';
      this.password = '';
    },(error) => {
      Swal.fire({
        text: 'Email et/ou Mot de passe incorrect(s)',
        icon: 'error',
        confirmButtonColor: '#3085d6',
        confirmButtonText: 'OK'
      });
    }
  );
}

register() {
var ok=true;
  let user2: any = {
    cin: this.cin,
    dateBirth: this.bd,
    name: this.nom,
    surname: this.surname,
    email: this.email,
    username: this.username2,
    password: this.password2,
    repassword:this.repassword2,
    tel:this.tel
  };


if( user2.cin.length!=8){
ok=false;

}
if(user2.repassword!=user2.password){
this.ok=false;
console.log("repassword!=password")
}



if(this.ok==true){
  this.authService.signup(user2).subscribe(
    (data) => {
      localStorage.setItem("user",   this.data)
      console.log( this.data)
        window.location.href = 'http://localhost:4200/d';
      user2 = {};
    },

    (err) => {
    
        Swal.fire({
          text: 'Email et/ou Mot de passe incorrect(s)',
          icon: 'error',
          confirmButtonColor: '#3085d6',
          confirmButtonText: 'OK'
        });
    }
  );
}else{
  Swal.fire({
    text: 'invalid form',
    icon: 'error',
    confirmButtonColor: '#3085d6',
    confirmButtonText: 'OK'
  });
}

}
}
