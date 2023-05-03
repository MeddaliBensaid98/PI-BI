import { Component } from '@angular/core';
import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import {  OnInit } from '@angular/core';

import { Observable } from 'rxjs';
import { map, shareReplay } from 'rxjs/operators';
import { AuthService } from '../services/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit{

  user:any;

   isHandset$: Observable<boolean> = this.breakpointObserver.observe(Breakpoints.Handset)
    .pipe(
      map(result => result.matches),
      shareReplay()
    );

  constructor(private breakpointObserver: BreakpointObserver,private router: Router,private authesrvice: AuthService) {  ;}

  ngOnInit() {

    if ((localStorage.getItem('user')==null) )
    {

      alert(' FORBIDDEN ACCESS  !!');
      this.router.navigate(['/']);

   }



    const userString = localStorage.getItem('user');
    if (userString !== null) {
      this.user = JSON.parse(userString);
    }
    this.authesrvice.getUser(this.user.username).subscribe(
      data => {
        this.user=data;


      },

      err => console.log(err));  }





  logout() {

  localStorage.removeItem('user');
  localStorage.clear()
  this.router.navigate(['']);

}

}
